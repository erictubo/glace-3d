import logging
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, sampler
from torch.utils.tensorboard import SummaryWriter

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
        # self.optimizer = Adam(
        #     filter(lambda p: p.requires_grad, self.encoder.parameters()),
        #     lr=options.learning_rate,
        # )
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=options.learning_rate,
            weight_decay=self.options.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        # Gradient scaler
        self.scaler = GradScaler(enabled=self.options.use_half)


        # Tensorboard logger
        config = {
            'output_path': options.output_path,

            'learning_rate': options.learning_rate,
            'weight_decay': options.weight_decay,

            'batch_size': options.batch_size,
            'gradient_accumulation_samples': options.gradient_accumulation_samples,
            'validation_frequency': options.validation_frequency,

            'use_half': options.use_half,
            'image_resolution': options.image_resolution,
            'aug_rotation': options.aug_rotation,
            'aug_scale_min': options.aug_scale_min,
            'aug_scale_max': options.aug_scale_max,

            'contrastive_weights': options.contrastive_weights,
            'train_dataset_size': len(self.train_dataset),
            'val_dataset_size': len(self.val_dataset),
        }

        self.logger = TensorBoardLogger(log_dir='runs/experiment_1', config=config)

        self.epoch = 0
        self.iteration = 0
        self.training_start = None

        # Validation
        self.encoder.eval()
        val_loss, val_loss_dict = self._validate('mse')
        self.logger.log_validation_epoch(val_loss_dict, self.epoch)

        _logger.info(
            f'Initial Validation, '
            f'Loss: {val_loss:.6f}'
        )


    def _load_datasets(self):
        
        train_dataset = RealFakeDataset(
            root_dir=self.options.data_path + "/train",
            augment=True,
            use_half=self.options.use_half,
            image_height=self.options.image_height,
        )

        val_dataset = RealFakeDataset(
            root_dir=self.options.data_path + "/validation",
            augment=False,
            use_half=self.options.use_half,
            image_height=self.options.image_height,
        )

        return train_dataset, val_dataset

    """
    TRAINING & VALIDATION
    """
    
    def train(self, num_epochs):

        _logger.info(f"Starting training ...")
        self.training_start = time.time()


        best_val_loss = float('inf')

        while self.epoch < num_epochs:

            self.epoch += 1

            if self.epoch <= num_epochs // 2:
                loss_type = 'mse'
            else:
                loss_type = 'cosine

            self.encoder.train()
            train_loss = self._train_epoch(loss_type)

            
            self.encoder.eval()
            val_loss, val_loss_dict = self._validate(loss_type)
            self.logger.log_validation_epoch(val_loss_dict, self.epoch)

            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.writer.add_scalar('learning_rate', current_lr, self.epoch)

            
            _logger.info(f'Epoch [{self.epoch}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _logger.info(f"Saving best model at epoch {self.epoch}")
                self.save_model(f"{self.options.output_path.split('.')[0]}_e{self.epoch}.pt")
            else:
                _logger.info(f"Stopping training because validation loss did not improve")
                break
        
        self.logger.close()
        
    def _train_epoch(self, loss_type):

        total_loss = 0.0
        accumulated_loss = 0.0
        accumulated_loss_dict = {}

        for real_images, fake_images, diff_images in self.train_loader:

            self.iteration += 1

            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)
            diff_images = diff_images.to(self.device)

            loss, loss_dict = self._compute_combined_loss(real_images, fake_images, diff_images, mode='training', loss_type=loss_type)
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

                _logger.info(f'Iteration {self.iteration}, Loss: {accumulated_loss:.6f}')
                accumulated_loss = 0.0

                if self.iteration % (self.options.gradient_accumulation_steps * self.options.validation_frequency) == 0:
                    val_loss, val_loss_dict = self._validate(loss_type)
                    self.logger.log_validation_iteration(val_loss_dict, self.iteration)
                    _logger.info(f'Iteration {self.iteration}, Val Loss: {val_loss:.6f}')

                    # Save model
                    _logger.info(f"Saving model at iteration {self.iteration}")
                    self.save_model(f"{self.options.output_path.split('.')[0]}_i{self.iteration}.pt")
                        
        return total_loss / len(self.train_loader)
    
    def _validate(self, loss_type):
        
        total_loss = 0.0
        accumulated_loss_dict = {}

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_images, fake_images, diff_images in self.val_loader:

                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)
                diff_images = diff_images.to(self.device)

                loss, loss_dict = self._compute_combined_loss(real_images, fake_images, diff_images, mode='validation', loss_type=loss_type)
                total_loss += loss.item()

                # Accumulate losses
                for key, value in loss_dict.items():
                    accumulated_loss_dict[key] = accumulated_loss_dict.get(key, 0) + value
            
        # Calculate average losses
        avg_loss_dict = {k: v / len(self.val_loader) for k, v in accumulated_loss_dict.items()}

        return total_loss / len(self.val_loader), avg_loss_dict

    """
    LOSS FUNCTIONS
    """

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
    
    # def _compute_separate_loss(self, real_image, fake_image, diff_image, mode, loss_type):
    #     """
    #     Loss for separate encoder to get fake_features close to real_init_features.
    #     """

    #     assert not torch.all(torch.eq(fake_image, diff_image)), "Negative samples are the same as positive samples"

    #     assert mode in ['training', 'validation']

    #     with autocast(enabled=self.options.use_half):

    #         with torch.no_grad():
    #             real_init_features = self.initial_encoder(real_image)

    #         with torch.no_grad() if mode=='validation' else torch.enable_grad():

    #             fake_features = self.encoder(fake_image)
    #             diff_features = self.encoder(diff_image)

    #             # Magnitudes
    #             fake_magnitude = self.magnitude_loss(fake_features)
    #             diff_magnitude = self.magnitude_loss(diff_features)

    #         magnitudes = fake_magnitude + diff_magnitude


    #         # MSE loss
    #         fake_vs_init_mse = self.mse_loss(fake_features, real_init_features)
    #         fake_vs_diff_mse = self.mse_loss(fake_features, diff_features, target_value=1.0)

    #         # Cosine loss
    #         fake_vs_init_cos = self.cosine_loss(fake_features, real_init_features, target_value=1, margin=0.2)
    #         fake_vs_diff_cos = self.cosine_loss(fake_features, diff_features, target_value=-1, margin=0.3)


    #         a, b = 1.0, 0.0
            
    #         if loss_type == 'mse':
    #             contrastive_loss = a * fake_vs_init_mse + b * fake_vs_diff_mse
    #         elif loss_type == 'cosine':
    #             contrastive_loss = a * fake_vs_init_cos + b * fake_vs_diff_cos
            
    #         loss = contrastive_loss + magnitudes


    #         loss_dict = {
    #             'F-I_cos': fake_vs_init_cos.item(),
    #             'F-D_cos': fake_vs_diff_cos.item(),

    #             'F-I_mse': fake_vs_init_mse.item(),
    #             'F-D_mse': fake_vs_diff_mse.item(),

    #             '|F|': fake_magnitude.item(),
    #             '|D|': diff_magnitude.item(),

    #             'Total': loss.item(),
    #         }

    #         return loss, loss_dict
    
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

            
            a, b, c = self.options.contrastive_weights

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
            self.data_path = "/home/johndoe/Documents/data/Transfer Learning/Pantheon"
            self.output_path = "output_encoder/fine-tuned_encoder_no_color_aug.pt"

            self.learning_rate = 0.0005
            self.weight_decay = 0.01

            self.batch_size = 4
            self.gradient_accumulation_samples = 40
            self.validation_frequency = 10

            self.use_half = True
            self.image_height = 480
            self.aug_rotation = 15
            self.aug_scale_min = 2/3
            self.aug_scale_max = 3/2

            self.contrastive_weights = (0.5, 0.25, 0.25)


    logging.basicConfig(level=logging.INFO)

    options = Options()
    trainer = TrainerEncoder(options)
    trainer.train(num_epochs=4)
    trainer.save_model(options.output_path)
