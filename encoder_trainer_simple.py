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
    real_images = [item[0] for item in batch]
    fake_images = [item[1] for item in batch]
    
    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images])
    max_width = max([img.shape[2] for img in real_images])
    
    # Pad images
    real_images_padded = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in real_images]
    fake_images_padded = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in fake_images]
    
    return torch.stack(real_images_padded), torch.stack(fake_images_padded)


class TrainerEncoder:
    def __init__(self, options):
        self.options = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")

        # Encoder to be fine-tuned
        self.encoder = Encoder()
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.to(self.device)

        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")


        # Encoder to be used for comparison
        self.initial_encoder = Encoder()
        self.initial_encoder.load_state_dict(encoder_state_dict)
        self.initial_encoder.to(self.device)
        self.initial_encoder.eval()

        for param in self.initial_encoder.parameters():
            param.requires_grad = False

        
        # Initialize dataset
        self.train_dataset, self.val_dataset = self._load_datasets()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.options.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
        

        _logger.info(f"Loaded training and validation datasets")

        # Initialize optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=options.learning_rate,
        )

        self.scaler = GradScaler(enabled=self.options.use_half)
        
        # Initialize loss function
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        self.encoder.eval()
        val_loss = self._validate()
        _logger.info(f'Initial Val Loss: {val_loss:.6f}')

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

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.encoder.train()
            train_loss = self._train_epoch()
            
            self.encoder.eval()
            val_loss = self._validate()
            
            _logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _logger.info(f"Saving best model at epoch {epoch+1}")
                self.save_model(f"{self.options.output_path.split('.')[0]}_e{epoch+1}.pt")
            
            # Early stopping (optional)
            if self._early_stopping(val_loss):
                _logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    def _train_epoch(self):

        total_loss = 0.0

        for real_images, fake_images in self.train_loader:

            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)

            loss_combined, fake_real_loss, real_real_loss, fake_real_initial_loss, fake_fake_initial_loss, fake_initial_real_initial_loss \
                = self._compute_losses(real_images, fake_images, use_no_grad=False) / self.options.gradient_accumulation_steps

            loss = loss_combined

            self.scaler.scale(loss).backward()

            self.iteration += 1

            if self.iteration % self.options.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.options.gradient_accumulation_steps

            if self.iteration % 1 == 0:
                time_since_start = time.time() - self.training_start
                _logger.info(
                    f'Iter {self.iteration:6d}, '
                    f'Loss: {loss:.6f}, '
                    f'F-R: {fake_real_loss:.6f}, '
                    f'R-R: {real_real_loss:.6f}, '
                    f'F-RI: {fake_real_initial_loss:.6f}, '
                    f'F-FI: {fake_fake_initial_loss:.6f}, '
                    f'FI-RI: {fake_initial_real_initial_loss:.6f}, '
                    f'Time: {time_since_start:.2f}s'
                )
            
            if self.iteration % 100 == 0:
                _logger.info(f'Saving model at iteration {self.iteration}')
                self.save_model(f"{self.options.output_path.split('.')[0]}_iter{self.iteration}.pt")
                self.encoder.eval()
                val_losses = self._validate()
                _logger.info(
                    f'Validation Iter {self.iteration:6d}, '
                    f'Loss: {val_losses[0]:.6f}, '
                    f'F-R: {val_losses[1]:.6f}, '
                    f'R-R: {val_losses[2]:.6f}, '
                    f'F-RI: {val_losses[3]:.6f}, '
                    f'F-FI: {val_losses[4]:.6f}, '
                    f'FI-RI: {val_losses[5]:.6f}'
                )
                        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        total_loss = 0.0
        total_fake_real_loss = 0.0
        total_real_real_loss = 0.0
        total_fake_real_initial_loss = 0.0
        total_fake_fake_initial_loss = 0.0
        total_fake_initial_real_initial_loss = 0.0

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_images, fake_images in self.val_loader:

                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)

                loss_combined, fake_real_loss, real_real_loss, fake_real_initial_loss, fake_fake_initial_loss, fake_initial_real_initial_loss \
                      = self._compute_loss(real_images, fake_images, use_no_grad=True)

                total_combined_loss += loss_combined.item()
                total_fake_real_loss += fake_real_loss.item()
                total_real_real_loss += real_real_loss.item()
                total_fake_real_initial_loss += fake_real_initial_loss.item()
                total_fake_fake_initial_loss += fake_fake_initial_loss.item()
                total_fake_initial_real_initial_loss += fake_initial_real_initial_loss.item()
            
            total_combined_loss /= len(self.val_loader)
            total_fake_real_loss /= len(self.val_loader)
            total_real_real_loss /= len(self.val_loader)
            total_fake_real_initial_loss /= len(self.val_loader)
            total_fake_fake_initial_loss /= len(self.val_loader)
            total_fake_initial_real_initial_loss /= len(self.val_loader)
        
        return total_combined_loss, total_fake_real_loss, total_real_real_loss, total_fake_real_initial_loss, total_fake_fake_initial_loss, total_fake_initial_real_initial_loss
    
    def _compute_losses(self, real_images, fake_images, use_no_grad):

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                real_initial_features = self.initial_encoder(real_images)
                fake_initial_features = self.initial_encoder(fake_images)

                real_initial_features = real_initial_features.view(real_initial_features.size(0), -1)
                fake_initial_features = fake_initial_features.view(fake_initial_features.size(0), -1)

            with torch.no_grad() if use_no_grad else torch.enable_grad():

                real_features = self.encoder(real_images)
                fake_features = self.encoder(fake_images)

                real_features = real_features.view(real_features.size(0), -1)
                fake_features = fake_features.view(fake_features.size(0), -1)

                target = torch.ones(fake_features.size(0)).to(self.device)

                fake_vs_real = self.cosine_loss(fake_features, real_features, target)
                real_vs_real_initial = self.cosine_loss(real_features, real_initial_features, target)

                w = self.options.loss_weight
                loss_combined = w * fake_vs_real + (1-w) * real_vs_real_initial

                fake_vs_real_initial = self.cosine_loss(fake_features, real_initial_features, target)
                fake_vs_fake_initial = self.cosine_loss(fake_features, fake_initial_features, target)
                fake_initial_vs_real_initial = self.cosine_loss(fake_initial_features, real_initial_features, target)

        return loss_combined, fake_vs_real, real_vs_real_initial, fake_vs_real_initial, fake_vs_fake_initial, fake_initial_vs_real_initial

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
            self.output_path = "output_encoder/fine_tuned_encoder.pt"

            self.learning_rate = 0.0005
            self.loss_weight = 0.7
            self.use_half = True
            self.image_resolution = 480
            self.use_aug = True
            self.aug_rotation = 15
            self.aug_scale = 1.5
            self.batch_size = 4
            self.gradient_accumulation_steps = 5

    logging.basicConfig(level=logging.INFO)

    options = Options()
    trainer = TrainerEncoder(options)
    trainer.train(num_epochs=10)
    trainer.save_model(options.output_path)
