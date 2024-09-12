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
        self.criterion = nn.MSELoss()

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
                self.save_model(f"fine_tuned_encoder_e{epoch+1}.pt")
            
            # Early stopping (optional)
            if self._early_stopping(val_loss):
                _logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    def _train_epoch(self):

        total_loss = 0.0

        for real_images, fake_images in self.train_loader:

            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)

            with torch.no_grad():
                real_initial_features = self.initial_encoder(real_images)

            with autocast(enabled=self.options.use_half):
                real_features = self.encoder(real_images)
                fake_features = self.encoder(fake_images)

                # Loss function for new fake encoder
                loss_separate = self.criterion(fake_features, real_initial_features)
                
                # Loss function for combined encoder
                loss_fake_real = self.criterion(fake_features, real_features)
                loss_real_initial = self.criterion(real_features, real_initial_features)
                w = 0.5
                combined_loss = w * loss_fake_real + (1-w) * loss_real_initial

            loss = loss_separate

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            self.iteration += 1

            if self.iteration % 1 == 0:
                # Print status.
                time_since_start = time.time() - self.training_start
                _logger.info(f'Iter {self.iteration:6d}, '
                 f'Separate Loss: {loss_separate:.6f}, '
                 f'Fake-Real Loss: {loss_fake_real:.6f}, '
                 f'Real-Initial Loss: {loss_real_initial:.6f}, '
                #  f'Combined Loss: {loss:.6f}, '
                 f'Time: {time_since_start:.2f}s')
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        total_loss = 0.0
        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_images, fake_images in self.val_loader:
                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)
                
                real_features = self.encoder(real_images)
                fake_features = self.encoder(fake_images)

                loss = self.criterion(real_features, fake_features)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def _early_stopping(self, val_loss):
        # Implement early stopping logic here
        # For example, stop if validation loss hasn't improved for X epochs
        pass
        
    def save_model(self, path):
        torch.save(self.encoder.state_dict(), path)


# All options:
class Options:
    def __init__(self):
        self.learning_rate = 0.002
        self.encoder_path = "ace_encoder_pretrained.pt"
        self.data_path = "/home/johndoe/Documents/data/Transfer Learning"
        self.use_half = True
        self.image_resolution = 480
        self.use_aug = True
        self.aug_rotation = 15
        self.aug_scale = 1.5
        self.batch_size = 8

# Usage
logging.basicConfig(level=logging.INFO)

options = Options()  # Assuming you have an Options class or similar configuration
trainer = TrainerEncoder(options)
trainer.train(num_epochs=10)
trainer.save_model("fine_tuned_encoder.pt")