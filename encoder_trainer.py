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


from ace_network import Encoder
from encoder_dataset_new import RealFakeDataset, custom_collate

_logger = logging.getLogger(__name__)


class TensorBoardLogger:
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
    Trainer class for fine-tuning the encoder network.
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

        
        # Dataset
        self.val_dataset, self.train_dataset, weights = self._load_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.options.batch_size,
            # shuffle=True,
            sampler=WeightedRandomSampler(
                weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            ),
            collate_fn=custom_collate,
        )
        # self.val_loader = DataLoader(
        #     self.val_dataset,
        #     batch_size=self.options.batch_size,
        #     shuffle=False,
        #     collate_fn=custom_collate,
        # )
        
        _logger.info(f"Loaded training and validation datasets")

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
            end_factor=0.1,
            total_iters=self.options.num_epochs,
        )

        # Gradient scaler
        self.scaler = GradScaler(enabled=self.options.use_half)


        # Tensorboard logger
        config = {
            'output_path': options.output_path,

            'datasets': options.dataset_names,
            'train_dataset_size': len(self.train_dataset),
            'val_dataset': options.validation_dataset,
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
        train_datasets = []
        for dataset_name in self.options.dataset_names:
            dataset_path = Path(self.options.data_path) / dataset_name

            if dataset_name == self.options.validation_dataset:
                val_dataset = RealFakeDataset(
                    root_dir=dataset_path,
                    augment=False,
                    use_half=self.options.use_half,
                    image_height=self.options.image_height,
                )
            else:
                train_datasets.append(RealFakeDataset(
                    root_dir=dataset_path,
                    augment=True,
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

        # _logger.info(f"Initial validation ...")

        self.save_model(f"{self.options.output_path}_e{self.epoch}.pt")

        # Validation
        self.encoder.eval()
        val_loss, val_loss_dict = self._validate('mse', self.options.epoch_val_limit)
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

            # Update negative samples
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

            loss_type = 'mse'
            # if self.epoch <= self.options.num_epochs // 2:
            #     loss_type = 'mse'
            # else:
            #     loss_type = 'cosine'

            self.encoder.train()
            train_loss = self._train_epoch(loss_type)

            
            self.encoder.eval()
            val_loss, val_loss_dict = self._validate(loss_type, self.options.epoch_val_limit)
            self.logger.log_validation_epoch(val_loss_dict, self.epoch)

            # Adjust learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.writer.add_scalar('learning_rate', current_lr, self.epoch)

            _logger.info(f'Epoch [{self.epoch}/{self.options.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (type: {loss_type})')
            
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
        
    def _train_epoch(self, loss_type):

        total_loss = 0.0
        accumulated_loss = 0.0
        accumulated_loss_dict = {}

        for image_mask, real_image, fake_image, diff_image, distance in self.train_loader:

            self.iteration += 1

            real_image = real_image.to(self.device)
            fake_image = fake_image.to(self.device)
            diff_image = diff_image.to(self.device)
            image_mask = image_mask.to(self.device)

            loss, loss_dict = self._compute_loss(real_image, fake_image, diff_image, image_mask, mode='training', loss_type=loss_type)
            accumulated_loss += loss
            total_loss += loss.item()

            for key, value in loss_dict.items():
                accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value

            loss /= self.options.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

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
                _logger.info(f'Iteration {self.iteration}/{len(self.train_loader)} Loss: {accumulated_loss:.6f} (type: {loss_type})')
                accumulated_loss = 0.0

                if self.iteration % (self.options.gradient_accumulation_steps * self.options.validation_frequency) == 0:
                    val_loss, val_loss_dict = self._validate(loss_type, self.options.iter_val_limit)
                    self.logger.log_validation_iteration(val_loss_dict, self.iteration)
                    _logger.info(f'Validation Loss: {val_loss:.6f}')

                    # Save model
                    _logger.info(f"Saving model at iteration {self.iteration}")
                    self.save_model(f"{self.options.output_path}_i{self.iteration}.pt")
            
            else:
                _logger.info(f'Iteration {self.iteration}/{len(self.train_loader)} ...')
                        
        return total_loss / len(self.train_loader)

    
    def get_random_validation_subset(self, n_samples):
        val_limit = min(n_samples, len(self.val_dataset))
        indices = torch.randperm(len(self.val_dataset))[:val_limit]
        subset = torch.utils.data.Subset(self.val_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )

    
    def _validate(self, loss_type, n_samples):

        _logger.info(f"Validating on {n_samples} samples ...")
        
        total_loss = 0.0
        accumulated_loss_dict = {}
        val_iteration = 0

        val_loader = self.get_random_validation_subset(n_samples)

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for image_mask, real_image, fake_image, diff_image, distance in val_loader:
                
                val_iteration += 1
                _logger.info(f'{val_iteration} / {len(val_loader)} ...')

                real_image = real_image.to(self.device)
                fake_image = fake_image.to(self.device)
                diff_image = diff_image.to(self.device)
                image_mask = image_mask.to(self.device)

                loss, loss_dict = self._compute_loss(real_image, fake_image, diff_image, image_mask, mode='validation', loss_type=loss_type)
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

    # def magnitude_loss(self, features, target_value=1.0, margin=0.1):

    #     feature_norms = torch.norm(features, dim=1)
        
    #     losses = F.relu(torch.abs(target_value - feature_norms) - margin)

    #     return losses.mean()
    
    def magnitude_loss(self, features, target_value=1.0, margin=0.15):

        magnitude = torch.mean(torch.norm(features, dim=1))

        return F.relu(torch.abs(target_value - magnitude) - margin)


    def mse_loss(self, features_1, features_2, target_value=0.0, margin=0.0):

        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)

        mse = F.mse_loss(features_1, features_2, reduction='none')

        losses = F.relu(torch.abs(target_value - mse) - margin)

        return losses.mean()
    

    def cosine_loss(self, features_1, features_2, target_value=1, margin=0.1):

        cos_sim = F.cosine_similarity(features_1, features_2, dim=1)

        losses = F.relu(torch.abs(target_value - cos_sim) - margin)

        return losses.mean()
    
    def _diversity_loss(self, features):
        
        # Encourage diversity in feature space
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.t())
        eye = torch.eye(features.size(0), device=features.device)

        return torch.mean((similarity_matrix - eye) ** 2)

    def _spatial_consistency_loss(self, features, image_mask):

        assert len(features.shape) == 4, features.shape
        B, C, H, W = features.shape

        mask_B1HW = image_mask
        mask_BCHW = mask_B1HW.expand(-1, C, -1, -1)

        assert mask_BCHW.shape == (B, C, H, W), mask_BCHW.shape

        # Encourage spatial consistency
        
        # Compute gradients in x and y directions
        grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
        grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]

        # Update mask to exclude border pixels (where gradients are not valid)
        mask = mask_BCHW[:, :, 1:, :] & mask_BCHW[:, :, :-1, :] & mask_BCHW[:, :, :, 1:] & mask_BCHW[:, :, :, :-1]

        # Mask gradients
        grad_x = grad_x[mask[:, :, :, 1:]]
        grad_y = grad_y[mask[:, :, 1:, :]]

        # Compute total variation
        return torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

    def triplet_loss(self, anchor, positive, negative, margin=0.2):

        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2) # positive, negative

        losses = F.relu(distance_positive - distance_negative + margin)

        return losses.mean()
    
    @staticmethod
    def _mask_features(features_list, image_mask):
        """
        Mask features to valid values only.
        """

        B, C, H, W = features_list[0].shape

        for features in features_list:
            assert features.shape == (B, C, H, W), features.shape


        feature_mask = TF.resize(image_mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)
        feature_mask = feature_mask.bool()

        assert feature_mask.shape == (B, 1, H, W), feature_mask.shape

        assert feature_mask.sum() != 0, "Mask is invalid everywhere!"

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
            print('M=N so mean feature mask should be equal to 1.0:')
            try: print(feature_mask.float().mean())
            except: print('error in mean calculation')
        

        assert(len(valid_features_list) == len(features_list))

        return valid_features_list


    def _compute_separate_loss(self, real_image, fake_image, diff_image, image_mask, mode, loss_type):
        """
        Loss for separate encoder to get fake_features close to real_init_features.
        """

        # assert not torch.all(torch.eq(fake_image, diff_image)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_init_features = self.initial_encoder(real_image)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                fake_features = self.encoder(fake_image)
                diff_features = self.encoder(diff_image)


            # Mask features
            real_init_features_NC, fake_features_NC, diff_features_NC = self._mask_features([real_init_features, fake_features, diff_features], image_mask)


            # Magnitudes
            real_init_magnitude = self.magnitude_loss(real_init_features_NC)
            fake_magnitude = self.magnitude_loss(fake_features_NC)
            diff_magnitude = self.magnitude_loss(diff_features_NC)

            magnitudes = 0.5 * fake_magnitude + 0.5 * diff_magnitude


            # MSE loss
            fake_vs_real_init_mse = 200* self.mse_loss(fake_features_NC, real_init_features_NC)
            # fake_vs_diff_mse = 200* self.mse_loss(fake_features_NC, diff_features_NC, target_value=1.0)

            # Cosine loss
            fake_vs_real_init_cos = self.cosine_loss(fake_features_NC, real_init_features_NC, target_value=1, margin=0.2)
            # fake_vs_diff_cos = self.cosine_loss(fake_features_NC, diff_features_NC, target_value=-1, margin=0.3)


            a, b = 1.0, 0.0
            
            if loss_type == 'mse':
                contrastive_loss = a * fake_vs_real_init_mse # + b * fake_vs_diff_mse
            elif loss_type == 'cosine':
                contrastive_loss = a * fake_vs_real_init_cos # + b * fake_vs_diff_cos
            
            loss = contrastive_loss + magnitudes


            loss_dict = {
                'F-I_cos': fake_vs_real_init_cos.item(),
                # 'F-D_cos': fake_vs_diff_cos.item(),

                'F-I_mse': fake_vs_real_init_mse.item(),
                # 'F-D_mse': fake_vs_diff_mse.item(),

                '|F|': fake_magnitude.item(),
                # '|D|': diff_magnitude.item(),
                '|RI|': real_init_magnitude.item(),

                'Total': loss.item(),
            }

            return loss, loss_dict
    
    def _compute_combined_loss(self, real_image, fake_image, diff_image, image_mask, mode, loss_type):
        """
        Contrastive loss function + magnitude loss.
        """

        assert not torch.all(torch.eq(fake_image, diff_image)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_init_features = self.initial_encoder(real_image)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                real_features = self.encoder(real_image)
                fake_features = self.encoder(fake_image)
                diff_features = self.encoder(diff_image)


            # Mask features
            real_init_features_NC, real_features_NC, fake_features_NC, diff_features_NC = self._mask_features([real_init_features, real_features, fake_features, diff_features], image_mask)


            # Magnitudes
            real_init_magnitude = self.magnitude_loss(real_init_features)
            real_magnitude = self.magnitude_loss(real_features_NC)
            fake_magnitude = self.magnitude_loss(fake_features_NC)
            diff_magnitude = self.magnitude_loss(diff_features_NC)

            magnitudes = real_magnitude + fake_magnitude + diff_magnitude


            # Cosine loss
            real_vs_fake_cos = self.cosine_loss(real_features_NC, fake_features_NC, target_value=1, margin=0.1)
            fake_vs_diff_cos = self.cosine_loss(fake_features_NC, diff_features_NC, target_value=-1, margin=0.3)
            real_vs_init_cos = self.cosine_loss(real_features_NC, real_init_features_NC, target_value=1, margin=0.2)

            # MSE loss
            real_vs_fake_mse = 200* self.mse_loss(real_features_NC, fake_features_NC)
            fake_vs_diff_mse = 1 - 200* self.mse_loss(fake_features_NC, diff_features_NC)
            real_vs_init_mse = 200* self.mse_loss(real_features_NC, real_init_features_NC) # TODO: add margin or remove

            
            a, b, c = self.options.contrastive_weights

            if loss_type == 'mse':
                contrastive_loss = a * real_vs_fake_mse + b * fake_vs_diff_mse + c * real_vs_init_mse

            elif loss_type == 'cosine':
                contrastive_loss = a * real_vs_fake_cos + b * fake_vs_diff_cos + c * real_vs_init_cos

            loss = contrastive_loss + magnitudes


            loss_dict = {
                'R-F_cos': real_vs_fake_cos.item(),
                'F-D_cos': fake_vs_diff_cos.item(),
                'R-RI_cos': real_vs_init_cos.item(),

                'R-F_mse': real_vs_fake_mse.item(),
                'F-D_mse': fake_vs_diff_mse.item(),
                'R-RI_mse': real_vs_init_mse.item(),
                
                '|RI|': real_init_magnitude.item(),
                '|R|': real_magnitude.item(),
                '|F|': fake_magnitude.item(),
                '|D|': diff_magnitude.item(),

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

            self.dataset_names = ['notre dame', 'brandenburg gate', 'pantheon']
            #self.validation_dataset = 'pantheon'
            self.iter_val_limit = 20 # number of samples for each validation
            self.epoch_val_limit = 80 # for epoch validation

            # self.output_path = "output_encoder/fine-tuned_encoder_separate.pt"
            # self.experiment_name = 'separate 1'


            self.learning_rate = 0.0005
            self.weight_decay = 0.01

            self.num_epochs = 4
            self.batch_size = 4
            self.gradient_accumulation_samples = 20
            self.validation_frequency = 5

            self.use_half = True
            self.image_height = 480
            self.aug_rotation = 40
            self.aug_scale_min = 240/480
            self.aug_scale_max = 960/480

            self.loss_function = 'combined'
            self.contrastive_weights = (0.5, 0.3, 0.2)


    logging.basicConfig(level=logging.INFO)

    options = Options()


    # def cross_validate():
    #     """
    #     Run configurations through all datasets and return best configuration.
    #     """

        # val_losses = []

        # for val_dataset in options.dataset_names:

        #     options.validation_dataset = val_dataset

    options.validation_dataset = 'brandenburg gate'

    w1, w2, w3 = options.contrastive_weights
    options.experiment_name = f"{options.loss_function}_w{w1}_{w2}_{w3}_{options.validation_dataset}"
    options.output_path = f"output_encoder/{options.experiment_name}"

    print(f'Training {options.experiment_name}')
    trainer = TrainerEncoder(options)
    val_loss = trainer.train()

    #         val_losses.append(val_loss)

    #     mean_val_loss = np.mean(val_losses)
    #     std_val_loss = np.std(val_losses)

    #     print(f"Mean validation losses:")
    #     print(mean_val_loss)

    #     print(f"Standard deviation of validation losses:")
    #     print(std_val_loss)

                
    #     return mean_val_loss, std_val_loss


    # mean_val_loss, std_val_loss = cross_validate()
