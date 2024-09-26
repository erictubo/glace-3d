import logging
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, sampler
import torch.distributed as dist

from ace_network import Encoder
from encoder_dataset import RealFakeDataset

_logger = logging.getLogger(__name__)

def custom_collate(batch):
    """
    Custom collate function to pad images to the same size.
    """
    real_images = [item[0] for item in batch]
    fake_images = [item[1] for item in batch]
    
    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images])
    max_width = max([img.shape[2] for img in real_images])
    
    # Pad images
    real_images_padded = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in real_images]
    fake_images_padded = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in fake_images]
    
    return torch.stack(real_images_padded), torch.stack(fake_images_padded)

def custom_collate_with_negatives(batch):
    """
    Custom collate function to pad images to the same size and include negative samples.
    """
    real_images = [item[0] for item in batch]
    fake_images = [item[1] for item in batch]
    
    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images + fake_images])
    max_width = max([img.shape[2] for img in real_images + fake_images])
    
    # Pad images
    def pad_images(images):
        return [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in images]
    
    real_images_padded = pad_images(real_images)
    fake_images_padded = pad_images(fake_images)

    # Create negative samples by shuffling fake images
    negative_images_padded = fake_images_padded.copy()
    random.shuffle(negative_images_padded)

    # Ensure negatives are different from positives
    for i in range(len(fake_images_padded)):
        if torch.all(torch.eq(fake_images_padded[i], negative_images_padded[i])):
            j = (i + 1) % len(fake_images_padded)  # Choose the next image as negative
            negative_images_padded[i], negative_images_padded[j] = negative_images_padded[j], negative_images_padded[i]
    
    
    return (torch.stack(real_images_padded), 
            torch.stack(fake_images_padded), 
            torch.stack(negative_images_padded))


class LossLogger:
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.train_losses = {}
        self.val_losses = {}
        self.train_iteration = 0
        self.val_iteration = 0

    def log(self, loss_dict, mode='training'):
        if mode == 'training':
            losses = self.train_losses
            self.train_iteration += 1
            iteration = self.train_iteration
        else:
            losses = self.val_losses
            self.val_iteration += 1
            iteration = self.val_iteration

        for key, value in loss_dict.items():
            if key not in losses:
                losses[key] = 0
            losses[key] += value

        if iteration % self.log_interval == 0:
            mean_losses = {k: v / self.log_interval for k, v in losses.items()}
            log_str = f'{"Train" if mode == "training" else "Val"} Iter {iteration:6d}, '
            log_str += ', '.join([f'{k}: {v:.6f}' for k, v in mean_losses.items()])
            _logger.info(log_str)
            if mode == 'training':
                self.train_losses = {}
            else:
                self.val_losses = {}

    def reset_val(self):
        self.val_losses = {}
        self.val_iteration = 0


class TrainerEncoder:
    """
    Trainer class for fine-tuning the encoder network.
    """

    def __init__(self, options):
        self.options = options
        self.options.gradient_accumulation_steps = options.gradient_accumulation_samples // options.batch_size
        
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
        self.train_dataset, self.val_dataset = self._load_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.options.batch_size,
            shuffle=True,
            collate_fn=custom_collate_with_negatives,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=custom_collate_with_negatives,
        )
        
        _logger.info(f"Loaded training and validation datasets")


        # Optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=options.learning_rate,
        )

        # Gradient scaler
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Loss logger
        self.loss_logger = LossLogger(log_interval=self.options.gradient_accumulation_steps)


        # Validation
        self.encoder.eval()
        val_loss = self._validate('mse')

        _logger.info(
            f'Initial Validation, '
            f'Loss: {val_loss:.6f}'
        )

        self.iteration = 0
        self.training_start = None

    def _load_datasets(self):
        ds_args = dict(
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
        )
        
        train_dataset = RealFakeDataset(
            root_dir=self.options.data_path + "/train",
            **ds_args,
        )

        val_dataset = RealFakeDataset(
            root_dir=self.options.data_path + "/validation",
            **ds_args,
        )

        return train_dataset, val_dataset
    
    def train(self, num_epochs):

        _logger.info(f"Starting training ...")
        self.training_start = time.time()

        loss_type = 'mse'

        best_val_loss = float('inf')
        for epoch in range(num_epochs):

            if epoch == num_epochs // 2:
                loss_type = 'cosine'
                _logger.info(f"Switching to cosine loss at epoch {epoch+1}")

            self.encoder.train()
            train_loss = self._train_epoch(loss_type)
            
            self.encoder.eval()
            val_loss = self._validate(loss_type)
            
            _logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _logger.info(f"Saving best model at epoch {epoch+1}")
                self.save_model(f"{self.options.output_path.split('.')[0]}_e{epoch+1}.pt")
            
            # Early stopping (optional)
            if self._early_stopping(val_loss):
                _logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    def _train_epoch(self, loss_type):

        total_loss = 0.0

        for real_images, fake_images, other_fake_images in self.train_loader:

            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)
            other_fake_images = other_fake_images.to(self.device)

            loss = self._compute_combined_loss(real_images, fake_images, other_fake_images, mode='training', loss_type=loss_type)

            loss /= self.options.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            self.iteration += 1

            if self.iteration % self.options.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.options.gradient_accumulation_steps

            # if self.iteration % 1 == 0:
            #     time_since_start = time.time() - self.training_start
            #     _logger.info(
            #         f'Iter {self.iteration:6d}, '
            #         f'Loss: {loss:.6f}, '
            #         f'Time: {time_since_start:.2f}s'
            #     )

            # if self.iteration % 100 == 0:
            #     _logger.info(f'Saving model at iteration {self.iteration}')
            #     self.save_model(f"{self.options.output_path.split('.')[0]}_iter{self.iteration}.pt")
            #     self.encoder.eval()
            #     val_los = self._validate()
            #     _logger.info(
            #         f'Validation Iter {self.iteration:6d},
            #         f'Loss: {val_loss:.6f}'
            #     )

                        
        return total_loss / len(self.train_loader)
    
    def _validate(self, loss_type):
        
        self.loss_logger.reset_val()

        total_loss = 0.0

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_images, fake_images, diff_images in self.val_loader:

                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)
                diff_images = diff_images.to(self.device)

                loss = self._compute_combined_loss(real_images, fake_images, diff_images, mode='validation', loss_type=loss_type)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def magnitude_loss(self, features, target_value=1.0, margin=0.05):
        
        return F.relu(torch.abs(target_value - torch.mean(torch.norm(features, dim=1))) - margin)
    
    def mse_loss(self, features_1, features_2, target_value=0.0, margin=0.0):

        # Normalize features
        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)

        return F.relu(torch.abs(target_value - F.mse_loss(features_1, features_2)) - margin)
    
    def cosine_loss(self, features_1, features_2, target_value=1, margin=0.1):

        # Flatten features
        features_1 = features_1.view(features_1.size(0), -1)
        features_2 = features_2.view(features_2.size(0), -1)

        target = target_value * torch.ones(features_1.size(0)).to(self.device)

        return F.cosine_embedding_loss(features_1, features_2, target=target, margin=margin)
    
    def triplet_loss(self, anchor, positive, negative, margin=0.2):

        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2) # positive, negative
        losses = F.relu(distance_positive - distance_negative + margin)

        return losses.mean()
    
    def _compute_separate_loss(self, real_image, fake_image, diff_image, mode, loss_type):
        """
        Loss for separate encoder to get fake_features close to real_init_features.
        """

        assert not torch.all(torch.eq(fake_image, diff_image)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_init_features = self.initial_encoder(real_image)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                fake_features = self.encoder(fake_image)
                diff_features = self.encoder(diff_image)

                # Magnitudes
                fake_magnitude = self.magnitude_loss(fake_features)
                diff_magnitude = self.magnitude_loss(diff_features)

            magnitudes = fake_magnitude + diff_magnitude

            # MSE loss
            fake_vs_init_mse = self.mse_loss(fake_features, real_init_features)
            fake_vs_diff_mse = self.mse_loss(fake_features, diff_features, target_value=1.0)

            # Cosine loss
            fake_vs_init_cos = self.cosine_loss(fake_features, real_init_features, target_value=1, margin=0.2)
            fake_vs_diff_cos = self.cosine_loss(fake_features, diff_features, target_value=-1, margin=0.3)

            a, b = 1.0, 0.0
            
            if loss_type == 'mse':
                contrastive_loss = a * fake_vs_init_mse + b * fake_vs_diff_mse
            elif loss_type == 'cosine':
                contrastive_loss = a * fake_vs_init_cos + b * fake_vs_diff_cos
            
            loss = contrastive_loss + magnitudes


            loss_dict = {
                'F-I_cos': fake_vs_init_cos.item(),
                'F-D_cos': fake_vs_diff_cos.item(),

                'F-I_mse': fake_vs_init_mse.item(),
                'F-D_mse': fake_vs_diff_mse.item(),

                '|F|': fake_magnitude.item(),
                '|D|': diff_magnitude.item(),

                'Total': loss.item(),
            }

            self.loss_logger.log(loss_dict, mode)

            return loss
    
    def _compute_combined_loss(self, real_image, fake_image, diff_image, mode, loss_type):
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

                # Magnitudes
                real_magnitude = self.magnitude_loss(real_features)
                fake_magnitude = self.magnitude_loss(fake_features)
                diff_magnitude = self.magnitude_loss(diff_features)

            magnitudes = real_magnitude + fake_magnitude + diff_magnitude


            # Cosine loss
            real_vs_fake_cos = self.cosine_loss(real_features, fake_features, target_value=1, margin=0.1)
            fake_vs_diff_cos = self.cosine_loss(fake_features, diff_features, target_value=-1, margin=0.3)
            real_vs_init_cos = self.cosine_loss(real_features, real_init_features, target_value=1, margin=0.2)

            # MSE loss
            real_vs_fake_mse = self.mse_loss(real_features, fake_features)
            fake_vs_diff_mse = self.mse_loss(fake_features, diff_features, target_value=1.0)
            real_vs_init_mse = self.mse_loss(real_features, real_init_features)

            
            a, b, c = 0.5, 0.2, 0.3

            if loss_type == 'mse':
                contrastive_loss = a * real_vs_fake_mse + b * fake_vs_diff_mse + c * real_vs_init_mse

            elif loss_type == 'cosine':
                contrastive_loss = a * real_vs_fake_cos + b * fake_vs_diff_cos + c * real_vs_init_cos

            loss = contrastive_loss + magnitudes


            loss_dict = {
                'R-F_cos': real_vs_fake_cos.item(),
                'F-D_cos': fake_vs_diff_cos.item(),
                'R-I_cos': real_vs_init_cos.item(),

                'R-F_mse': real_vs_fake_mse.item(),
                'F-D_mse': fake_vs_diff_mse.item(),
                'R-I_mse': real_vs_init_mse.item(),

                '|R|': real_magnitude.item(),
                '|F|': fake_magnitude.item(),
                '|D|': diff_magnitude.item(),

                'Total': loss.item(),
            }

            # TODO: logging on tensorboard
            
            self.loss_logger.log(loss_dict, mode)

        return loss

    def _early_stopping(self, val_loss):
        # Implement early stopping logic here
        # For example, stop if validation loss hasn't improved for X epochs
        pass
        
    def save_model(self, path):
        torch.save(self.encoder.state_dict(), path)


if __name__ == "__main__":

    class Options:
        def __init__(self):
            self.encoder_path = "ace_encoder_pretrained.pt"
            self.data_path = "/home/johndoe/Documents/data/Transfer Learning/Pantheon"
            self.output_path = "output_encoder/fine-tuned_encoder_no_color_aug.pt"
            self.learning_rate = 0.0005
            self.contrastive_weights = (0.4, 0.3, 0.3)
            # self.loss_weight = 0.5

            self.use_half = True
            self.image_resolution = 480
            self.use_aug = True
            self.aug_rotation = 15
            self.aug_scale = 1.5
            self.batch_size = 4
            self.gradient_accumulation_samples = 40

    logging.basicConfig(level=logging.INFO)

    options = Options()
    trainer = TrainerEncoder(options)
    trainer.train(num_epochs=4)
    trainer.save_model(options.output_path)
