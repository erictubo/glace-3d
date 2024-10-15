import logging
import random
import time
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import functional as TF
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter


from ace_network import Encoder, Head
from encoder_dataset import RealFakeDataset, custom_collate

_logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    Logging of training and validation losses to TensorBoard.
    """
    def __init__(self, log_dir, config):
        self.writer = SummaryWriter(log_dir)
        
        # Log configuration
        for key, value in config.items():
            self.writer.add_text(f"config/{key}", str(value))

    def log_train(self, loss_dict, iteration, epoch):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'train/{key}', value, iteration)
        self.writer.add_scalar('epoch', epoch, iteration)

    def log_validation_epoch(self, loss_dict, epoch):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'validation_epoch/{key}', value, epoch)

    def log_validation_iteration(self, loss_dict, iteration):
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'validation/{key}', value, iteration)

    def close(self):
        self.writer.close()


class TrainerEncoder:
    """
    Fine-tuning of a pre-trained encoder network to bridge the domain gap between real and fake images.
    """

    def __init__(self, options):
        options.gradient_accumulation_steps = options.gradient_accumulation_samples // options.batch_size

        if options.output_path.endswith('.pt'):
            options.output_path = options.output_path[:-3]

        self.options = options
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")

        # Encoder to be fine-tuned
        self.encoder = Encoder()
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.to(self.device)

        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")


        # Encoder (static) to be used for reference
        self.initial_encoder = Encoder()
        self.initial_encoder.load_state_dict(encoder_state_dict)
        self.initial_encoder.to(self.device)
        self.initial_encoder.eval()

        for param in self.initial_encoder.parameters():
            param.requires_grad = False

        # Create head for each dataset
        self.heads = {}
        for dataset_name, head_path in self.options.head_paths.items():
            head_state_dict = torch.load(head_path, map_location="cpu")
            self.heads[dataset_name] = Head.create_from_state_dict(head_state_dict)
            self.heads[dataset_name].to(self.device)

            for param in self.heads[dataset_name].parameters():
                param.requires_grad = False
        
            _logger.info(f"Loaded prediction head for {dataset_name} from: {head_path}")

        # Dataset
        self.val_dataset, self.train_dataset, weights = self._load_datasets()
        _logger.info(f"Loaded training and validation datasets")

        # Sampler to represent datasets equally
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(self.train_dataset),
            replacement=True, # can repeat samples
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.options.batch_size,
            # shuffle=True, # not compatible with sampler
            sampler=sampler,
            collate_fn=custom_collate,
        )
        # self.val_loader = DataLoader(
        #     self.val_dataset,
        #     batch_size=self.options.batch_size,
        #     shuffle=False,
        #     collate_fn=custom_collate,
        # )

        # Loss function
        if self.options.loss_function == 'separate':
            self._compute_loss = self._compute_separate_loss
        elif self.options.loss_function == 'combined':
            self._compute_loss = self._compute_combined_loss
        else:
            raise ValueError(f"Invalid loss function: {self.options.loss_function}")

        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=self.options.learning_rate,
            weight_decay=self.options.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.2,
            total_iters=self.options.num_epochs * int(len(self.train_dataset) / self.options.gradient_accumulation_steps),
        )

        # Gradient scaler
        self.scaler = GradScaler(enabled=self.options.use_half)


        # Tensorboard logger
        config = {
            'output_path': options.output_path,

            'datasets': options.dataset_names,
            'train_dataset_size': len(self.train_dataset),
            'val_dataset': options.val_dataset_name,
            'val_dataset_size': len(self.val_dataset),

            'learning_rate': options.learning_rate,
            'weight_decay': options.weight_decay,

            'num_epochs' : options.num_epochs,
            'batch_size': options.batch_size,
            'gradient_accumulation_samples': options.gradient_accumulation_samples,

            'use_half': options.use_half,
            'image_height': options.image_height,
            'aug_rotation': options.aug_rotation,
            'aug_scale_min': options.aug_scale_min,
            'aug_scale_max': options.aug_scale_max,
            
            'loss_function': options.loss_function,
            'contrastive_weights': options.contrastive_weights,
        }

        self.logger = TensorBoardLogger(
            log_dir='runs/' + options.experiment_name,
            config=config,
        )

        self.epoch = 0
        self.iteration = 0
        self.training_start = None
        

    def _load_datasets(self):
        """
        Load training and validation datasets.
        """
        train_datasets = []
        for dataset_name in self.options.dataset_names:
            dataset_path = Path(self.options.data_path) / dataset_name

            if dataset_name == self.options.val_dataset_name:
                val_dataset = RealFakeDataset(
                    root_dir=dataset_path,
                    name = dataset_name,
                    augment=False,
                    use_half=self.options.use_half,
                    image_height=self.options.image_height,
                )
            else:
                train_datasets.append(RealFakeDataset(
                    root_dir=dataset_path,
                    name = dataset_name,
                    augment=True,
                    augment_color=True,
                    use_half=self.options.use_half,
                    image_height=self.options.image_height,
                ))

        train_dataset = ConcatDataset(train_datasets)

        dataset_sizes = [len(dataset) for dataset in train_datasets]
        weights = []

        for dataset_size in dataset_sizes:
            weights.extend([1 / dataset_size] * dataset_size)

        weights = torch.DoubleTensor(weights)

        return val_dataset, train_dataset, weights

    """
    TRAINING & VALIDATION
    """
    
    def train(self):

        # Make sure saving the model works
        self.save_model(f"{self.options.output_path}_e{self.epoch}.pt")

        # Initial validation
        self.encoder.eval()
        val_loss, val_loss_dict = self._validate(self.options.epoch_val_limit)
        self.logger.log_validation_epoch(val_loss_dict, self.epoch)
        self.logger.log_validation_iteration(val_loss_dict, self.iteration)

        _logger.info(f'Validation Loss: {val_loss:.6f}')

        best_val_loss = val_loss
        patience = 2
        patience_counter = 0

        _logger.info(f"Starting training ...")
        self.training_start = time.time()

        while self.epoch < self.options.num_epochs:

            self.epoch += 1

            # Update negative samples for each epoch
            if isinstance(self.train_dataset, ConcatDataset):
                for dataset in self.train_dataset.datasets:
                    dataset.set_epoch(self.epoch)
            else:
                self.train_dataset.set_epoch(self.epoch)
                
            if isinstance(self.val_dataset, ConcatDataset):
                for dataset in self.val_dataset.datasets:
                    dataset.set_epoch(self.epoch)
            else:
                self.val_dataset.set_epoch(self.epoch)


            self.encoder.train()
            train_loss = self._train_epoch()

            
            self.encoder.eval()
            val_loss, val_loss_dict = self._validate(self.options.epoch_val_limit)
            self.logger.log_validation_epoch(val_loss_dict, self.epoch)

            _logger.info(f'Epoch [{self.epoch}/{self.options.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _logger.info(f"Saving best model at epoch {self.epoch}")
                self.save_model(f"{self.options.output_path}_e{self.epoch}.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                _logger.info(f"Stopping training because validation loss did not improve")
                break
        
        self.logger.close()
        return best_val_loss
        
    def _train_epoch(self):

        total_loss = 0.0
        accumulated_loss = 0.0
        accumulated_loss_dict = {}

        for real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name in self.train_loader:

            self.iteration += 1

            real_image_1 = real_image_1.to(self.device)
            real_image_2 = real_image_2.to(self.device)
            fake_image_1 = fake_image_1.to(self.device)
            fake_image_2 = fake_image_2.to(self.device)
            mask_1 = mask_1.to(self.device)
            mask_2 = mask_2.to(self.device)
            coords_1 = coords_1.to(self.device)
            coords_2 = coords_2.to(self.device)
            glob_1 = glob_1.to(self.device)
            glob_2 = glob_2.to(self.device)

            loss, loss_dict = self._compute_loss(
                real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name, mode='training')
            accumulated_loss += loss
            total_loss += loss.item()

            for key, value in loss_dict.items():
                accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value

            loss /= self.options.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if self.iteration % self.options.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                total_norm = self._compute_gradient_norm()
                self.logger.writer.add_scalar('gradient_norm', total_norm, self.iteration)

                # Log average losses
                avg_loss_dict = {k: v / self.options.gradient_accumulation_steps for k, v in accumulated_loss_dict.items()}
                self.logger.log_train(avg_loss_dict, self.iteration, self.epoch)
                accumulated_loss_dict = {}

                accumulated_loss /= self.options.gradient_accumulation_steps
                _logger.info(f'Iteration {self.iteration}/{len(self.train_loader)} Loss: {accumulated_loss:.6f}')
                accumulated_loss = 0.0

                if self.iteration % (self.options.gradient_accumulation_steps * self.options.validation_frequency) == 0:
                    val_loss, val_loss_dict = self._validate(self.options.iter_val_limit)
                    self.logger.log_validation_iteration(val_loss_dict, self.iteration)
                    _logger.info(f'Validation Loss: {val_loss:.6f}')

                    # Save model
                    _logger.info(f"Saving model at iteration {self.iteration}")
                    self.save_model(f"{self.options.output_path}_i{self.iteration}.pt")
                
                # Adjust learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.writer.add_scalar('learning_rate', current_lr, self.iteration)
            
            else:
                _logger.info(f'Iteration {self.iteration}/{len(self.train_loader)} ...')

            # Stop training if maximum number of iterations reached
            if self.iteration >= self.options.max_iterations:
                _logger.info(f"Stopping training because maximum number of iterations reached")
                break
                        
        return total_loss / len(self.train_loader)

    def get_random_validation_subset(self, n_samples):
        """
        Choose a random subset of the validation dataset.
        """
        val_limit = min(n_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:val_limit]
        subset = torch.utils.data.Subset(self.val_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
    
    def _validate(self, n_samples):

        _logger.info(f"Validating on {n_samples} samples ...")
        
        total_loss = 0.0
        accumulated_loss_dict = {}
        val_iteration = 0

        val_loader = self.get_random_validation_subset(n_samples)

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name in val_loader:
                
                val_iteration += 1
                _logger.info(f'{val_iteration} / {len(val_loader)} ...')

                real_image_1 = real_image_1.to(self.device)
                real_image_2 = real_image_2.to(self.device)
                fake_image_1 = fake_image_1.to(self.device)
                fake_image_2 = fake_image_2.to(self.device)
                mask_1 = mask_1.to(self.device)
                mask_2 = mask_2.to(self.device)
                coords_1 = coords_1.to(self.device)
                coords_2 = coords_2.to(self.device)
                glob_1 = glob_1.to(self.device)
                glob_2 = glob_2.to(self.device)

                loss, loss_dict = self._compute_loss(
                    real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name,  mode='validation')
                total_loss += loss.item()

                # Accumulate losses
                for key, value in loss_dict.items():
                    accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value
            
        # Calculate average losses
        avg_loss_dict = {k: v / len(val_loader) for k, v in accumulated_loss_dict.items()}

        return total_loss / len(val_loader), avg_loss_dict

    """
    LOSS FUNCTIONS
    """
    
    def _magnitude_loss(self, features, target_value=1.0, margin=0.15):

        magnitude = torch.mean(torch.norm(features, p=2, dim=1))

        return F.relu(torch.abs(target_value - magnitude) - margin)
    
        # feature_norms = torch.norm(features, dim=1)
        # losses = F.relu(torch.abs(target_value - feature_norms) - margin)

        # return losses.mean()

    # def _mse_loss(self, features_1, features_2, target_value=0.0, margin=0.0, p=1):

    #     features_1 = F.normalize(features_1, p=p, dim=1)
    #     features_2 = F.normalize(features_2, p=p, dim=1)

    #     mse = F.mse_loss(features_1, features_2, reduction='none')

    #     losses = F.relu(torch.abs(target_value - mse) - margin)

    #     return losses.mean()
    
    # def _mae_loss(self, features_1, features_2, target_value=0.0, margin=0.0, smooth=True, p=1):

    #     features_1 = F.normalize(features_1, p=p, dim=1)
    #     features_2 = F.normalize(features_2, p=p, dim=1)

    #     if smooth:
    #         mae = F.smooth_l1_loss(features_1, features_2, reduction='none')
    #     else:
    #         mae = F.l1_loss(features_1, features_2, reduction='none')

    #     losses = F.relu(torch.abs(target_value - mae) - margin)

    #     return losses.mean()
    
    def _cosine_loss(self, features_1, features_2, target_value=1, margin=0.1):

        cos_sim = F.cosine_similarity(features_1, features_2, dim=1)

        losses = F.relu(torch.abs(target_value - cos_sim) - margin)

        return losses.mean()
    
    # def _diversity_loss(self, features, feature_mask):
    #     B, C, H, W = features.shape
        
    #     # Reshape features
    #     features_reshaped = features.view(B, C, -1)  # Shape: (B, C, H*W)
        
    #     # Reshape mask_1
    #     mask_reshaped = feature_mask.view(B, 1, -1)  # Shape: (B, 1, H*W)
        
    #     # Apply mask_1
    #     features_masked = features_reshaped * mask_reshaped  # Broadcasting the mask_1
        
    #     # Normalize features
    #     features_norm = F.normalize(features_masked, p=2, dim=1)
        
    #     # Compute similarity matrix
    #     similarity_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
        
    #     # Compute diversity loss
    #     eye = torch.eye(C, device=features.device).unsqueeze(0).expand(B, -1, -1)
    #     diversity_loss = torch.mean((similarity_matrix - eye) ** 2)

    #     return diversity_loss

    # def _spatial_consistency_loss(self, features, feature_mask):

    #     assert len(features.shape) == 4, features.shape
    #     B, C, H, W = features.shape

    #     mask_B1HW = feature_mask
    #     mask_BCHW = mask_B1HW.expand(-1, C, -1, -1)

    #     assert mask_BCHW.shape == (B, C, H, W), mask_BCHW.shape
        
    #     # Compute gradients in x and y directions
    #     grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
    #     grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]

    #     # Update mask_1 to exclude border pixels (where gradients are not valid)
    #     mask_x = mask_BCHW[:, :, :, 1:] & mask_BCHW[:, :, :, :-1]
    #     mask_y = mask_BCHW[:, :, 1:, :] & mask_BCHW[:, :, :-1, :]

    #     # mask_1 gradients
    #     grad_x_masked = grad_x[mask_x]
    #     grad_y_masked = grad_y[mask_y]

    #     # Compute total variation
    #     return torch.mean(torch.abs(grad_x_masked)) + torch.mean(torch.abs(grad_y_masked))

    # def _triplet_loss(self, anchor, positive, negative, margin=0.2):

    #     distance_positive = F.pairwise_distance(anchor, positive, p=2)
    #     distance_negative = F.pairwise_distance(anchor, negative, p=2) # positive, negative

    #     losses = F.relu(distance_positive - distance_negative + margin)

    #     return losses.mean()

    @staticmethod
    def _resize_coords_to_features(coords, features_shape, visualize=False):
        """
        Resize coordinates to the same size as features, subsampling to each patch.
        Input: coords (shape Bx3xIHxIW where IWxIH is the image size), features_shape (BxCxHxW)\\
        Output: feature_coords (shape Bx3xHxW)
        """
            
        B, C, H, W = features_shape

        feature_coords = F.interpolate(coords, size=[H, W], mode='nearest')

        assert feature_coords.shape == (B, 3, H, W), feature_coords.shape

        if visualize:
            import matplotlib.pyplot as plt
            from encoder_dataset import coords_to_colors
            fig, ax = plt.subplots(2, 4)
            for i in range(B):
                ax[0, i].imshow(coords_to_colors(coords[i].cpu()))
                ax[1, i].imshow(coords_to_colors(feature_coords[i].cpu()))
                print('...')
            plt.show()

        return feature_coords
    
    @staticmethod
    def _resize_mask_to_features(image_mask, features_shape, threshold=0.25, visualize=False):
        """
        Resize mask to the same size as features, subsampling to each patch.
        Input: image_mask (shape Bx1xIHxIW where IWxIH is the image size), features_shape (BxCxHxW)\\
        Output: feature_mask (shape Bx1xHxW)
        """

        B, C, H, W = features_shape

        feature_mask = F.interpolate(image_mask.float(), size=[H, W], mode='area')
        feature_mask = (feature_mask >= threshold).bool()

        assert feature_mask.shape == (B, 1, H, W), feature_mask.shape

        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 4)
            for i in range(B):
                ax[0, i].imshow(image_mask[i][0].cpu(), cmap='gray')
                ax[1, i].imshow(feature_mask[i][0].cpu(), cmap='gray')
                print('...')
            plt.show()

        assert feature_mask.sum() != 0, "mask_1 is invalid everywhere!"

        return feature_mask
    
    # TODO: separate functions for visualization of coordinates & masks
    
    @staticmethod
    def _mask_features(features_list, feature_mask):
        """
        Mask features to valid values only, reshaping to 2D tensor.\\
        Input: features_list (shape BxCxHxW each), feature_mask (shape Bx1xHxW)\\
        Output: valid_features_list (shape MxC each, where M is the number of valid patches, M <= N = B*H*W)
        """

        B, C, H, W = features_list[0].shape

        for features in features_list:
            assert features.shape == (B, C, H, W), features.shape

        def normalize_shape(tensor_in):
            """Bring tensor from shape BxCxHxW to NxC"""
            return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)
        
        feature_mask_N1 = normalize_shape(feature_mask)

        assert feature_mask_N1.shape == (B*H*W, 1), feature_mask.shape

        feature_mask_NC = feature_mask_N1.expand(B*H*W, C)

        assert feature_mask_NC.shape == (B*H*W, C), feature_mask_NC.shape

        features_NC_list = [normalize_shape(features) for features in features_list]

        def apply_mask(features_NC, mask_NC):
            valid_features = features_NC[mask_NC]

            N, C = features_NC.shape

            return valid_features.reshape(-1, C)

        valid_features_list = [apply_mask(features_NC, feature_mask_NC) for features_NC in features_NC_list]

        M, C = valid_features_list[0].shape

        for valid_features in valid_features_list:
            assert valid_features.shape == (M, C), valid_features.shape
        
        N = B*H*W
        assert M <= N, f"Masked size {M} larger than {N} = {B}*{H}*{W}"
        if M == N:
            print('M=N so mean feature mask_1 should be equal to 1.0:')
            try: print(feature_mask.float().mean())
            except: print('error in mean calculation')
        
        assert(len(valid_features_list) == len(features_list))

        return valid_features_list, M
    
    # def compute_3d_norm(tensor1, tensor2):
    #     # Ensure the tensors have the same shape
    #     assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
        
    #     # Compute the difference
    #     diff = tensor1 - tensor2
        
    #     # Compute the squared norm along the coordinate dimension (dim=1)
    #     squared_norm = torch.sum(diff**2, dim=1)
        
    #     # Take the square root to get the Euclidean norm
    #     norm = torch.sqrt(squared_norm)
        
    #     return norm


    def _compute_separate_loss(self, real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name, mode):
        """
        Loss for training a separate encoder:
        A) make fake_features similar to real_init_features (real_vs_fake),
        B) keep fake_features_1 and fake_features_2 distinct (fake_1_vs_fake_2),
        + Maintain magntiude of features close to 1.0.
        """

        assert not torch.all(torch.eq(fake_image_1, fake_image_2)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_init_features_1 = self.initial_encoder(real_image_1)
                real_init_features_2 = self.initial_encoder(real_image_2)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                fake_features_1 = self.encoder(fake_image_1)
                fake_features_2 = self.encoder(fake_image_2)


            # SCENE COORDINATES
            gt_coords_1 = self._resize_coords_to_features(coords_1, fake_features_1.shape)
            gt_coords_2 = self._resize_coords_to_features(coords_2, fake_features_2.shape)

            glob_features_1 = glob_1[..., None, None].expand(-1, -1, fake_features_1.shape[2], fake_features_1.shape[3])
            glob_features_2 = glob_2[..., None, None].expand(-1, -1, fake_features_2.shape[2], fake_features_2.shape[3])

            glob_local_features_1 = torch.cat((glob_features_1, fake_features_1), dim=1)
            glob_local_features_2 = torch.cat((glob_features_2, fake_features_2), dim=1)
            
            with autocast(enabled=self.options.use_half):

                pred_coords_1 = self.heads[dataset_name](glob_local_features_1)
                pred_coords_2 = self.heads[dataset_name](glob_local_features_2)

            assert pred_coords_1.shape == gt_coords_1.shape, f"{pred_coords_1.shape} != {gt_coords_1.shape}"
            assert pred_coords_2.shape == gt_coords_2.shape, f"{pred_coords_2.shape} != {gt_coords_2.shape}"


            valid_coords_1 = (gt_coords_1.sum(dim=1) != 0)
            valid_coords_2 = (gt_coords_2.sum(dim=1) != 0)

            distance_1 = torch.norm(gt_coords_1 - pred_coords_1, p=2, dim=1)
            distance_2 = torch.norm(gt_coords_2 - pred_coords_2, p=2, dim=1)

            distance_1_valid = distance_1[valid_coords_1]
            distance_2_valid = distance_2[valid_coords_2]

            V1 = distance_1_valid.size(0)
            V2 = distance_2_valid.size(0)

            coords_loss = (V1 * distance_1_valid.mean() + V2 * distance_2_valid.mean()) / (V1 + V2)

            print(f'Coords loss: {round(coords_loss.item(), 2)}')

            def visualize_coords():

                difference_1 = distance_1.cpu().numpy()
                difference_2 = distance_2.cpu().numpy()
                valid_1 = valid_coords_1.cpu().numpy()
                valid_2 = valid_coords_2.cpu().numpy()
                masked_difference_1 = np.ma.masked_array(difference_1, mask=~valid_1)
                masked_difference_2 = np.ma.masked_array(difference_2, mask=~valid_2)

                import matplotlib.pyplot as plt
                from encoder_dataset import coords_to_colors

                cmap = plt.get_cmap('Spectral').reversed()
                cmap.set_bad(color='white')
                fig, ax = plt.subplots(self.options.batch_size, 6)
                for i in range(self.options.batch_size):
                    ax[i, 0].imshow(coords_to_colors(gt_coords_1[i].cpu()))
                    ax[i, 1].imshow(coords_to_colors(pred_coords_1[i].cpu()))
                    ax[i, 2].imshow(masked_difference_1[i], cmap=cmap)
                    ax[i, 3].imshow(coords_to_colors(gt_coords_2[i].cpu()))
                    ax[i, 4].imshow(coords_to_colors(pred_coords_2[i].cpu()))
                    ax[i, 5].imshow(masked_difference_2[i], cmap=cmap)
                    print('...')
                plt.show()


            # MASKING
            mask_1 = self._resize_mask_to_features(mask_1, fake_features_1.shape)
            mask_2 = self._resize_mask_to_features(mask_2, fake_features_2.shape)
            mask_combined = torch.logical_and(mask_1, mask_2)

            (real_init_features_1_MC, fake_features_1_MC), M = self._mask_features([real_init_features_1, fake_features_1], mask_1)
            (real_init_features_2_NC, fake_features_2_NC), N = self._mask_features([real_init_features_2, fake_features_2], mask_2)

            (real_init_features_1_OC, fake_features_1_OC, real_init_features_2_OC, fake_features_2_OC), O = \
                self._mask_features([real_init_features_1, fake_features_1, real_init_features_2, fake_features_2], mask_combined)

            # MAGNITUDE LOSS
            magnitude_real_1_init = self._magnitude_loss(real_init_features_1_MC)
            magnitude_real_2_init = self._magnitude_loss(real_init_features_2_NC)
            magnitude_real_init = 0.5 * (magnitude_real_1_init + magnitude_real_2_init)

            magnitude_fake_1 = self._magnitude_loss(fake_features_1_MC)
            magnitude_fake_2 = self._magnitude_loss(fake_features_2_NC)
            magnitude_fake = 0.5 * (magnitude_fake_1 + magnitude_fake_2)

            magnitude = magnitude_fake

            # COSINE LOSS
            # Minimize
            cosine_fake_1_vs_real_1_init = self._cosine_loss(fake_features_1_MC, real_init_features_1_MC, target_value=1, margin=0.1)
            cosine_fake_2_vs_real_2_init = self._cosine_loss(fake_features_2_NC, real_init_features_2_NC, target_value=1, margin=0.1)
            cosine_fake_vs_real_init = (M * cosine_fake_1_vs_real_1_init + N * cosine_fake_2_vs_real_2_init) / (M + N) # Equal weighting of each valid patch

            # Maximize
            cosine_fake_1_vs_fake_2 = self._cosine_loss(fake_features_1_OC, fake_features_2_OC, target_value=-1, margin=0.3)

            # Track
            cosine_real_1_init_vs_real_2_init = self._cosine_loss(real_init_features_1_OC, real_init_features_2_OC, target_value=-1, margin=0.3)

            # a, b = self.options.contrastive_weights

            # loss = a * (cosine_fake_vs_real_init) + b * (cosine_fake_1_vs_fake_2)

            # loss += magnitude

            loss = coords_loss

            loss_dict = {
                'F_3D' : coords_loss.item(),

                'F-Ri_cos': cosine_fake_vs_real_init.item(),
                'F1-F2_cos': cosine_fake_1_vs_fake_2.item(),
                'R1i-R2i_cos': cosine_real_1_init_vs_real_2_init.item(),

                '|F|': magnitude_fake.item(),
                '|Ri|': magnitude_real_init.item(),

                'Total': loss.item(),
            }

            return loss, loss_dict
    
    def _compute_combined_loss(self, real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, coords_1, coords_2, glob_1, glob_2, idx_1, idx_2, dataset_name, mode):
        """
        Loss for training a combined encoder:
        A) make fake_features similar to real_features (real_vs_fake),
        B) keep fake_features_1 and fake_features_2 distinct (fake_1_vs_fake_2),
        C) anchor real_features to real_init_features (real_vs_real_init),
        + Maintain magntiude of features close to 1.0.
        """

        assert not torch.all(torch.eq(fake_image_1, fake_image_2)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_init_features_1 = self.initial_encoder(real_image_1)
                real_init_features_2 = self.initial_encoder(real_image_2)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                real_features_1 = self.encoder(real_image_1)
                real_features_2 = self.encoder(real_image_2)
                fake_features_1 = self.encoder(fake_image_1)
                fake_features_2 = self.encoder(fake_image_2)

            # MASKING
            mask_1 = self._resize_mask_to_features(mask_1, fake_features_1.shape)
            mask_2 = self._resize_mask_to_features(mask_2, fake_features_2.shape)
            mask_combined = torch.logical_and(mask_1, mask_2)

            (real_init_features_1_MC, real_features_1_MC, fake_features_1_MC), M = self._mask_features([real_init_features_1, real_features_1, fake_features_1], mask_1)
            (real_init_features_2_NC, real_features_2_NC, fake_features_2_NC), N = self._mask_features([real_init_features_2, real_features_2, fake_features_2], mask_2)
            (fake_features_1_OC, fake_features_2_OC), O = self._mask_features([fake_features_1, fake_features_2], mask_combined)

            # MAGNITUDE
            # Optimize to 1.0
            magnitude_real_1 = self._magnitude_loss(real_features_1_MC)
            magnitude_real_2 = self._magnitude_loss(real_features_2_NC)
            magnitude_real = 0.5 * (magnitude_real_1 + magnitude_real_2)

            magnitude_fake_1 = self._magnitude_loss(fake_features_1_MC)
            magnitude_fake_2 = self._magnitude_loss(fake_features_2_NC)
            magnitude_fake = 0.5 * (magnitude_fake_1 + magnitude_fake_2)
            
            # Track
            magnitude_real_1_init = self._magnitude_loss(real_init_features_1_MC)
            magnitude_real_2_init = self._magnitude_loss(real_init_features_2_NC)
            magnitude_real_init = 0.5 * (magnitude_real_1_init + magnitude_real_2_init)

            magnitude = 0.5 * (magnitude_real + magnitude_fake)

            # COSINE LOSS
            # Minimize
            cosine_real_1_vs_fake_1 = self._cosine_loss(real_features_1_MC, fake_features_1_MC, target_value=1, margin=0.1)
            cosine_real_2_vs_fake_2 = self._cosine_loss(real_features_2_NC, fake_features_2_NC, target_value=1, margin=0.1)
            cosine_real_vs_fake = (M * cosine_real_1_vs_fake_1 + N * cosine_real_2_vs_fake_2) / (M + N) # Equal weighting of each valid patch

            # Maximize
            cosine_fake_1_vs_fake_2 = self._cosine_loss(fake_features_1_OC, fake_features_2_OC, target_value=-1, margin=0.3)
            
            # Anchor
            cosine_real_1_vs_real_1_init = self._cosine_loss(real_features_1_MC, real_init_features_1_MC, target_value=1, margin=0.2)
            cosine_real_2_vs_real_2_init = self._cosine_loss(real_features_2_NC, real_init_features_2_NC, target_value=1, margin=0.2)
            cosine_real_vs_real_init = (M * cosine_real_1_vs_real_1_init + N * cosine_real_2_vs_real_2_init) / (M + N)
            
            a, b, c = self.options.contrastive_weights

            loss = a * cosine_real_vs_fake + b * cosine_fake_1_vs_fake_2 + c * cosine_real_vs_real_init

            loss += magnitude

            loss_dict = {
                'R-F_cos': cosine_real_vs_fake.item(),
                'F1-F2_cos': cosine_fake_1_vs_fake_2.item(),
                'R-Ri_cos': cosine_real_vs_real_init.item(),

                '|Ri|': magnitude_real_init.item(),
                '|R|': magnitude_real.item(),
                '|F|': magnitude_fake.item(),

                'Total': loss.item(),
            }
        
        return loss, loss_dict
        
    def save_model(self, path):
        torch.save(self.encoder.state_dict(), path)

    def _compute_gradient_norm(self):
        total_norm = 0
        for p in self.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm


if __name__ == "__main__":

    class Options:
        def __init__(self):
            self.encoder_path = "ace_encoder_pretrained.pt"
            self.data_path = "/home/johndoe/Documents/data/Transfer Learning/"

            self.dataset_names = ['notre dame', 'pantheon'] #, 'brandenburg gate']

            self.head_paths = {
                'notre dame': 'output/notre_dame.pt',
                'pantheon': 'output/pantheon.pt',
                #'brandenburg gate': 'output/brandenburg_gate.pt',
            }

            self.iter_val_limit = 20 # number of samples for each validation
            self.epoch_val_limit = 80 # for epoch validation

            self.learning_rate = 0.0005
            self.weight_decay = 0.01

            self.max_iterations = 1250
            self.num_epochs = 2
            self.batch_size = 4
            self.gradient_accumulation_samples = 20
            self.validation_frequency = 5

            self.use_half = True
            self.image_height = 480
            self.aug_rotation = 40
            self.aug_scale_min = 240/480
            self.aug_scale_max = 960/480


    logging.basicConfig(level=logging.INFO)

    options = Options()


    # Test
    options.loss_function = 'separate'
    options.val_dataset_name = 'pantheon' # 'brandenburg gate'
    options.contrastive_weights = (0.0, 0.0)
    options.experiment_name = "test-coords"
    options.output_path = f"output_encoder/{options.experiment_name}"

    print(f'Training {options.experiment_name}')
    trainer = TrainerEncoder(options)
    val_loss = trainer.train()


    # Validation

    # for options.val_dataset_name in options.dataset_names:

    #     options.loss_function = 'separate'

    #     for options.contrastive_weights in [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)]:

    #         w1, w2 = options.contrastive_weights
    #         options.experiment_name = f"val_separate_w{w1}_{w2}_{options.val_dataset_name}"
    #         options.output_path = f"output_encoder/{options.experiment_name}"

    #         print(f'Training {options.experiment_name}')
    #         trainer = TrainerEncoder(options)
    #         val_loss = trainer.train()


    #     options.loss_function = 'combined'

    #     for options.contrastive_weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.4, 0.2, 0.4)]:

    #         w1, w2, w3 = options.contrastive_weights
    #         options.experiment_name = f"val_combined_w{w1}_{w2}_{w3}_{options.val_dataset_name}"
    #         options.output_path = f"output_encoder/{options.experiment_name}"

    #         print(f'Training {options.experiment_name}')
    #         trainer = TrainerEncoder(options)
    #         val_loss = trainer.train()


    # print('Finished')
