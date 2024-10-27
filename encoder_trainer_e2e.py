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

from ace_network import Encoder, Head, Regressor
from encoder_dataset import RealFakeDataset, custom_collate
from encoder_loss import coords_loss, magnitude_loss, cosine_loss, mask_features


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


class TrainerEncoderE2E:
    """
    End-to-end fine-tuning of a pre-trained encoder network to bridge the domain gap between real and fake images.
    """

    def __init__(self, options):
        options.gradient_accumulation_steps = options.gradient_accumulation_samples // options.batch_size

        if options.output_path.endswith('.pt'):
            options.output_path = options.output_path[:-3]

        self.options = options
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")

        # Single head / dataset
        head_state_dict = torch.load(self.options.head_path, map_location="cpu")

        # Encoder (static) to be used for reference
        self.initial_encoder = Encoder()
        self.initial_encoder.load_state_dict(encoder_state_dict)
        self.initial_encoder.to(self.device)
        self.initial_encoder.eval()

        for param in self.initial_encoder.parameters():
            param.requires_grad = False
        
        _logger.info(f"Loaded encoder from {self.options.encoder_path}")

        # End-to-end regressor = encoder + head
        self.regressor = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
        self.regressor.to(self.device)

        _logger.info(f"Loaded regressor from {self.options.head_path}")


        # Optimizer
        self.optimizer = AdamW(self.regressor.parameters(), lr=self.options.learning_rate)


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

        # Learning rate scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.options.num_epochs * int(len(self.train_dataset) / self.options.gradient_accumulation_steps),
        )

        # Gradient scaler
        self.scaler = GradScaler(enabled=self.options.use_half)


        # Loss function
        if self.options.loss_function == 'separate':
            self._compute_loss = self._compute_separate_loss
        elif self.options.loss_function == 'combined':
            self._compute_loss = self._compute_combined_loss
        else:
            raise ValueError(f"Invalid loss function: {self.options.loss_function}")


        # Tensorboard logger
        config = {
            'output_path': options.output_path,

            'datasets': options.dataset_names,
            'train_dataset_size': len(self.train_dataset),
            'val_dataset': options.val_dataset_name,
            'val_dataset_size': len(self.val_dataset),

            'learning_rate': options.learning_rate,
            # 'weight_decay': options.weight_decay,

            'num_epochs' : options.num_epochs,
            'batch_size': options.batch_size,
            'gradient_accumulation_samples': options.gradient_accumulation_samples,
            'clip_norm': options.clip_norm,

            'use_half': options.use_half,
            'image_height': options.image_height,
            'aug_rotation': options.aug_rotation,
            'aug_scale_min': options.aug_scale_min,
            'aug_scale_max': options.aug_scale_max,
            
            'loss_function': options.loss_function,
            # 'cosine_weights': options.cosine_weights,
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
        self.regressor.eval()
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

            self.regressor.train()
            train_loss = self._train_epoch()

            
            self.regressor.eval()
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

            if patience_counter >= patience or self.iteration >= self.options.max_iterations:
                _logger.info(f"Stopping training because validation loss did not improve")
                break
        
        self.logger.close()
        return best_val_loss
        
    def _train_epoch(self):

        total_loss = 0.0
        accumulated_loss = 0.0
        accumulated_loss_dict = {}

        with autocast(enabled=self.options.use_half):
            for batch in self.train_loader:

                self.iteration += 1

                # dataset_names = batch[-1]
                batch = (item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in batch)

                loss, loss_dict = self._compute_loss(batch, mode='training')
                accumulated_loss += loss
                total_loss += loss.item()

                for key, value in loss_dict.items():
                    accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value

                loss /= self.options.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                # # Check regressor gradients         
                # for name, param in self.regressor.named_parameters():
                #     if param.grad is None:
                #         print(f"No gradient for encoder: {name}")
                #     elif torch.isnan(param.grad).any():
                #         print(f"NaN gradient for encoder {name}")
                #     else:
                #         print(f"OK gradient for encoder {name}: {param.grad.norm().item()}")

                # TODO: multiple datasets

                # check that not all gradients are zero / NaN
                if all(param.grad is None or torch.isnan(param.grad).all() for param in self.regressor.parameters()):
                    print("All encoder gradients are None or NaN")
                    break

                # Gradient accumulation
                if self.iteration % self.options.gradient_accumulation_steps == 0:

                    # Clip gradients
                    total_norm = self._compute_gradient_norm()
                    self.logger.writer.add_scalar('gradient_norm', total_norm, self.iteration)

                    if self.options.clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=self.options.clip_norm)
                        print("Clipping gradients ...")
                        clipped_norm = self._compute_gradient_norm()
                        print(f"Norm before clipping: {total_norm:.6f}, after clipping: {clipped_norm:.6f}")
                        self.logger.writer.add_scalar('clipped_gradient_norm', clipped_norm, self.iteration)

                    self.optimizer.step()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.optimizer.zero_grad(set_to_none=True)

                    self.log_weight_changes()

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
            for batch in val_loader:
                
                val_iteration += 1
                _logger.info(f'{val_iteration} / {len(val_loader)} ...')

                batch = (item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in batch)

                loss, loss_dict = self._compute_loss(batch,  mode='validation')
                total_loss += loss.item()

                # Accumulate losses
                for key, value in loss_dict.items():
                    accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value
            
        # Calculate average losses
        avg_loss_dict = {k: v / len(val_loader) for k, v in accumulated_loss_dict.items()}

        return total_loss / len(val_loader), avg_loss_dict

    def _compute_separate_loss(self, batch, mode):
        """
        Loss for training a separate encoder for real (pre-trained) and fake images (fine-tuned):
        End-to-end training of encoder + head to predict accurate 3D coordinates.
        """

        real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, gt_fake_coords_1, gt_fake_coords_2, real_glob_1, real_glob_2, fake_glob_1, fake_glob_2, idx_1, idx_2, dataset_names = batch

        mask_combined = torch.logical_and(mask_1, mask_2)

        assert not torch.all(torch.eq(fake_image_1, fake_image_2)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with torch.no_grad():
            real_init_features_1 = self.initial_encoder(real_image_1)
            real_init_features_2 = self.initial_encoder(real_image_2)

        # with torch.no_grad() if mode=='validation' else torch.enable_grad():
        #     fake_features_1 = self.regressor.get_features(fake_image_1)
        #     fake_features_2 = self.regressor.get_features(fake_image_2)

        # # assert that mask shape and coords shape match features shape
        # assert mask_1.shape[2:] == gt_coords_1.shape[2:] == fake_features_1.shape[2:], f"{mask_1.shape} != {gt_coords_1.shape} != {fake_features_1.shape}"
        # assert mask_2.shape[2:] == gt_coords_2.shape[2:] == fake_features_2.shape[2:], f"{mask_2.shape} != {gt_coords_2.shape} != {fake_features_2.shape}"

        with torch.no_grad() if mode=='validation' else torch.enable_grad():
            with autocast(enabled=True):
                pred_fake_coords_1 = self.regressor(fake_image_1, fake_glob_1)
                pred_fake_coords_2 = self.regressor(fake_image_2, fake_glob_2)

        with torch.no_grad() if mode=='validation' else torch.enable_grad():
            # pred_fake_coords_1 = self._predict_coords(fake_features_1, fake_glob_1, dataset_names)
            # pred_fake_coords_2 = self._predict_coords(fake_features_2, fake_glob_2, dataset_names)

            # Separate: using pre-trained encoder for real images
            pred_real_coords_1 = self._predict_coords(real_init_features_1, real_glob_1, dataset_names)
            pred_real_coords_2 = self._predict_coords(real_init_features_2, real_glob_2, dataset_names)

        # # 3D LOSS
        fake_coords_1_loss, V1 = coords_loss(gt_fake_coords_1, pred_fake_coords_1, median=self.options.median)
        fake_coords_2_loss, V2 = coords_loss(gt_fake_coords_2, pred_fake_coords_2, median=self.options.median)
        fake_coords_loss = (V1 * fake_coords_1_loss + V2 * fake_coords_2_loss) / (V1 + V2)

        real_coords_1_loss, V3 = coords_loss(gt_fake_coords_1, pred_real_coords_1, median=self.options.median)
        real_coords_2_loss, V4 = coords_loss(gt_fake_coords_2, pred_real_coords_2, median=self.options.median)
        real_coords_loss = (V3 * real_coords_1_loss + V4 * real_coords_2_loss) / (V3 + V4)

        # loss = 0.5 * (fake_coords_loss + real_coords_loss) / 50
        loss = fake_coords_loss / 50


        # # MASKING
        # (real_init_features_1_MC, fake_features_1_MC), M = mask_features([real_init_features_1, fake_features_1], mask_1)
        # (real_init_features_2_NC, fake_features_2_NC), N = mask_features([real_init_features_2, fake_features_2], mask_2)

        # (real_init_features_1_OC, fake_features_1_OC, real_init_features_2_OC, fake_features_2_OC), O = \
        #     mask_features([real_init_features_1, fake_features_1, real_init_features_2, fake_features_2], mask_combined)


        # # MAGNITUDE LOSS
        # magnitude_real_1_init = magnitude_loss(real_init_features_1_MC)
        # magnitude_real_2_init = magnitude_loss(real_init_features_2_NC)
        # magnitude_real_init = 0.5 * (magnitude_real_1_init + magnitude_real_2_init)

        # magnitude_fake_1 = magnitude_loss(fake_features_1_MC)
        # magnitude_fake_2 = magnitude_loss(fake_features_2_NC)
        # magnitude_fake = 0.5 * (magnitude_fake_1 + magnitude_fake_2)

        # magnitude = magnitude_fake


        # # COSINE LOSS
        # # Minimize
        # cosine_fake_1_vs_real_1_init = cosine_loss(fake_features_1_MC, real_init_features_1_MC, target_value=1, margin=0.1)
        # cosine_fake_2_vs_real_2_init = cosine_loss(fake_features_2_NC, real_init_features_2_NC, target_value=1, margin=0.1)
        # cosine_fake_vs_real_init = (M * cosine_fake_1_vs_real_1_init + N * cosine_fake_2_vs_real_2_init) / (M + N) # Equal weighting of each valid patch

        # # Maximize
        # cosine_fake_1_vs_fake_2 = cosine_loss(fake_features_1_OC, fake_features_2_OC, target_value=-1, margin=0.3)

        # # Track
        # cosine_real_1_init_vs_real_2_init = cosine_loss(real_init_features_1_OC, real_init_features_2_OC, target_value=-1, margin=0.3)


        # a, b = self.options.cosine_weights

        # loss = a * (cosine_fake_vs_real_init) + b * (cosine_fake_1_vs_fake_2)


        # if self.options.magnitude:
        #     loss += magnitude


        loss_dict = {
            'F_3D' : fake_coords_loss.item(),
            'R_3D' : real_coords_loss.item(),

            # 'F-Ri_cos': cosine_fake_vs_real_init.item(),
            # 'F1-F2_cos': cosine_fake_1_vs_fake_2.item(),
            # 'R1i-R2i_cos': cosine_real_1_init_vs_real_2_init.item(),

            # '|F|': magnitude_fake.item(),
            # '|Ri|': magnitude_real_init.item(),

            'Total': loss.item(),
        }

        return loss, loss_dict
    
    def _compute_combined_loss(self, batch, mode):
        """
        Loss for training a combined encoder (real and fake images share the same fine-tuned encoder):
        End-to-end training of encoder + head to predict accurate 3D coordinates.
        """

        real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, gt_fake_coords_1, gt_fake_coords_2, real_glob_1, real_glob_2, fake_glob_1, fake_glob_2, idx_1, idx_2, dataset_names = batch

        mask_combined = torch.logical_and(mask_1, mask_2)
        
        assert not torch.all(torch.eq(fake_image_1, fake_image_2)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        # with torch.no_grad():
        #     real_init_features_1 = self.initial_encoder(real_image_1)
        #     real_init_features_2 = self.initial_encoder(real_image_2)

        with torch.no_grad() if mode=='validation' else torch.enable_grad():
            real_features_1 = self.regressor.get_features(real_image_1)
            real_features_2 = self.regressor.get_features(real_image_2)
            fake_features_1 = self.regressor.get_features(fake_image_1)
            fake_features_2 = self.regressor.get_features(fake_image_2)

        assert mask_1.shape[2:] == gt_fake_coords_1.shape[2:] == real_features_1.shape[2:], f"{mask_1.shape} != {gt_fake_coords_1.shape} != {real_features_1.shape}"
        assert mask_2.shape[2:] == gt_fake_coords_2.shape[2:] == real_features_2.shape[2:], f"{mask_2.shape} != {gt_fake_coords_2.shape} != {real_features_2.shape}"


        # 3D LOSS
        with torch.no_grad() if mode=='validation' else torch.enable_grad():
            pred_fake_coords_1 = self._predict_coords(fake_features_1, fake_glob_1, dataset_names)
            pred_fake_coords_2 = self._predict_coords(fake_features_2, fake_glob_2, dataset_names)

            # Combined: using same fine-tuned encoder
            pred_real_coords_1 = self._predict_coords(real_features_1, real_glob_1, dataset_names)
            pred_real_coords_2 = self._predict_coords(real_features_2, real_glob_2, dataset_names)

        fake_coords_1_loss, V1 = coords_loss(gt_fake_coords_1, pred_fake_coords_1, median=self.options.median)
        fake_coords_2_loss, V2 = coords_loss(gt_fake_coords_2, pred_fake_coords_2, median=self.options.median)
        fake_coords_loss = (V1 * fake_coords_1_loss + V2 * fake_coords_2_loss) / (V1 + V2)

        real_coords_1_loss, V3 = coords_loss(gt_fake_coords_1, pred_real_coords_1, median=self.options.median)
        real_coords_2_loss, V4 = coords_loss(gt_fake_coords_2, pred_real_coords_2, median=self.options.median)
        real_coords_loss = (V3 * real_coords_1_loss + V4 * real_coords_2_loss) / (V3 + V4)

        loss = 0.5 * (fake_coords_loss + real_coords_loss) / 50


        # # MASKING
        # (real_init_features_1_MC, real_features_1_MC, fake_features_1_MC), M = mask_features([real_init_features_1, real_features_1, fake_features_1], mask_1)
        # (real_init_features_2_NC, real_features_2_NC, fake_features_2_NC), N = mask_features([real_init_features_2, real_features_2, fake_features_2], mask_2)
        # (fake_features_1_OC, fake_features_2_OC), O = mask_features([fake_features_1, fake_features_2], mask_combined)


        # # MAGNITUDE
        # # Optimize to 1.0
        # magnitude_real_1 = magnitude_loss(real_features_1_MC)
        # magnitude_real_2 = magnitude_loss(real_features_2_NC)
        # magnitude_real = 0.5 * (magnitude_real_1 + magnitude_real_2)

        # magnitude_fake_1 = magnitude_loss(fake_features_1_MC)
        # magnitude_fake_2 = magnitude_loss(fake_features_2_NC)
        # magnitude_fake = 0.5 * (magnitude_fake_1 + magnitude_fake_2)
        
        # # Track
        # magnitude_real_1_init = magnitude_loss(real_init_features_1_MC)
        # magnitude_real_2_init = magnitude_loss(real_init_features_2_NC)
        # magnitude_real_init = 0.5 * (magnitude_real_1_init + magnitude_real_2_init)

        # magnitude = 0.5 * (magnitude_real + magnitude_fake)


        # # COSINE LOSS
        # # Minimize
        # cosine_real_1_vs_fake_1 = cosine_loss(real_features_1_MC, fake_features_1_MC, target_value=1, margin=0.1)
        # cosine_real_2_vs_fake_2 = cosine_loss(real_features_2_NC, fake_features_2_NC, target_value=1, margin=0.1)
        # cosine_real_vs_fake = (M * cosine_real_1_vs_fake_1 + N * cosine_real_2_vs_fake_2) / (M + N) # Equal weighting of each valid patch

        # # Maximize
        # cosine_fake_1_vs_fake_2 = cosine_loss(fake_features_1_OC, fake_features_2_OC, target_value=-1, margin=0.3)
        
        # # Anchor
        # cosine_real_1_vs_real_1_init = cosine_loss(real_features_1_MC, real_init_features_1_MC, target_value=1, margin=0.2)
        # cosine_real_2_vs_real_2_init = cosine_loss(real_features_2_NC, real_init_features_2_NC, target_value=1, margin=0.2)
        # cosine_real_vs_real_init = (M * cosine_real_1_vs_real_1_init + N * cosine_real_2_vs_real_2_init) / (M + N)
        

        # a, b, c = self.options.cosine_weights

        # loss = a * cosine_real_vs_fake + b * cosine_fake_1_vs_fake_2 + c * cosine_real_vs_real_init
        

        # if self.options.magnitude:
        #     loss += magnitude


        loss_dict = {
            'F_3D' : fake_coords_loss.item(),
            'R_3D' : real_coords_loss.item(),

            # 'R-F_cos': cosine_real_vs_fake.item(),
            # 'F1-F2_cos': cosine_fake_1_vs_fake_2.item(),
            # 'R-Ri_cos': cosine_real_vs_real_init.item(),

            # '|Ri|': magnitude_real_init.item(),
            # '|R|': magnitude_real.item(),
            # '|F|': magnitude_fake.item(),

            'Total': loss.item(),
        }
    
        return loss, loss_dict

    def _predict_coords(self, local_features, global_features, dataset_names):
        """
        Predict scene coordinates from local and global features by passing through the corresponding prediction head.
        """
        glob_features = global_features[..., None, None].expand(-1, -1, local_features.shape[2], local_features.shape[3])
        glob_local_features = torch.cat((glob_features, local_features), dim=1)

        # E2E single dataset, w/o validation
        return self.regressor.get_scene_coordinates(glob_local_features)

        # if all(name == dataset_names[0] for name in dataset_names):
        #     return self.heads[dataset_names[0]](glob_local_features)
        
        # pred_coords_list = []

        # for i, dataset_name in enumerate(dataset_names):
        #     pred_coords = self.heads[dataset_name](glob_local_features[i].unsqueeze(0))
        #     pred_coords_list.append(pred_coords.squeeze(0))
        
        # return torch.stack(pred_coords_list, dim=0)
    
    def save_model(self, path):
        torch.save(self.regressor.encoder.state_dict(), path)

    def _compute_gradient_norm(self, printing=False):
        total_norm = 0
        for name, param in self.regressor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if printing: print(f"Gradient norm for {name}: {param_norm:.6f}")
        total_norm = total_norm ** 0.5
        if printing: print(f"Total gradient norm: {total_norm:.6f}")
        return total_norm
    
    def log_weight_changes(self):
        if not hasattr(self, 'previous_weights'):
            self.previous_weights = {name: param.data.clone() for name, param in self.regressor.named_parameters()}
            return

        total_change = 0
        num_params = 0
        for name, param in self.regressor.named_parameters():
            change = torch.abs(param.data - self.previous_weights[name]).mean().item()
            total_change += change
            num_params += 1
            print(f"Average change for {name}: {change:.6f}")
            self.previous_weights[name] = param.data.clone()

        avg_change = total_change / num_params
        print(f"Overall average change in weights: {avg_change:.6f}")
        
        # Log to TensorBoard
        self.logger.writer.add_scalar('weight_change/avg', avg_change, self.iteration)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    class Options:
        def __init__(self):
            self.use_half = True
            self.image_height = 480
            self.aug_rotation = 40
            self.aug_scale_min = 240/480
            self.aug_scale_max = 960/480


            self.encoder_path = "ace_encoder_pretrained.pt"
            self.data_path = "/home/johndoe/Documents/data/Transfer Learning/"


            self.batch_size = 2
            self.num_epochs = 2
            self.max_iterations = 5000 / self.batch_size
            self.gradient_accumulation_samples = 20
            self.validation_frequency = 5

            self.iter_val_limit = 20 # number of samples for each validation
            self.epoch_val_limit = 80 # for epoch validation


            self.dataset_names = ['pantheon', 'pantheon test'] # 'brandenburg gate', 'notre dame']

            self.val_dataset_name = 'pantheon test' # 'brandenburg gate'

            self.head_path = 'output/pantheon.pt'
            # TODO: multiple heads for multiple datasets / requires splitting regressor into encoder + heads
            # self.head_paths = {
            #     'pantheon': 'output/pantheon.pt',
            #     'brandenburg gate': 'output/brandenburg_gate.pt',
            #     # 'notre dame': 'output/notre_dame.pt',
            # }


            self.learning_rate = 0.0005
            # self.weight_decay = 0.01

            self.loss_function = 'separate'
            self.median = False
            self.magnitude = False

            # self.cosine_weights = (0.0, 0.0)

    options = Options()


    # TESTING

    # for options.learning_rate in [0.003, 0.001, 0.0003,  0.0001, 0.00003]:
    options.learning_rate = 0.001
    options.max_iterations = 1000 / options.batch_size
    options.clip_norm = 3.0

    options.experiment_name = f"e2e-fake_clip-{options.clip_norm}_loss-{str(options.loss_function)}_lr-{str(options.learning_rate)}"
    # options.experiment_name = 'test'
    options.output_path = f"output_encoder/{options.experiment_name}"

    print(f'Training {options.experiment_name}')
    trainer = TrainerEncoderE2E(options)
    val_loss = trainer.train()


    # VALIDATION

    # for options.magnitude in (False, True):
    # for loss_name, options.median in [('mean', False), ('median', True)]:
    #     for options.val_dataset_name in options.dataset_names:


    #         options.loss_function = 'separate'

    #         # options.train_dataset_name = [name for name in options.dataset_names if name != options.val_dataset_name][0]
    #         # options.experiment_name = f"3d-{loss_name}_train-{train_dataset_name}_val-{options.val_dataset_name}"
    #         options.output_path = f"output_encoder/{options.experiment_name}"

    #         print(f'Training {options.experiment_name}')
    #         trainer = TrainerEncoder(options)
    #         # val_loss = trainer.train()

    #     for options.cosine_weights in [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4)]:

    #         w1, w2 = options.cosine_weights
    #         options.experiment_name = f"val_separate_w{w1}_{w2}_{options.val_dataset_name}"
    #         options.output_path = f"output_encoder/{options.experiment_name}"

    #         print(f'Training {options.experiment_name}')
    #         trainer = TrainerEncoder(options)
    #         val_loss = trainer.train()


    #     options.loss_function = 'combined'

    #     for options.cosine_weights in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.4, 0.2, 0.4)]:

    #         w1, w2, w3 = options.cosine_weights
    #         options.experiment_name = f"val_combined_w{w1}_{w2}_{w3}_{options.val_dataset_name}"
    #         options.output_path = f"output_encoder/{options.experiment_name}"

    #         print(f'Training {options.experiment_name}')
    #         trainer = TrainerEncoder(options)
    #         val_loss = trainer.train()


    # print('Finished')

