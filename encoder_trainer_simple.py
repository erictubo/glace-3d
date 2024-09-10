import logging
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, sampler
import torch.distributed as dist

from ace_network import Encoder
from encoder_dataset import RealFakeDataset


encoder_path = "ace_encoder_pretrained.pt"

class TrainerEncoder:
    def __init__(self, options):
        self.options = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize encoder
        encoder = Encoder()  # Assuming you have an Encoder class defined

        encoder_state_dict = torch.load(encoder_path, map_location="cpu")
        encoder.load_state_dict(encoder_state_dict)

        encoder.to(self.device)
        
        # Initialize dataset
        self.train_dataset, self.val_dataset = self._load_datasets()

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.options.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.options.batch_size, shuffle=False)

        # Initialize optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=options.learning_rate,
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()


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
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.encoder.train()
            train_loss = self._train_epoch()
            
            self.encoder.eval()
            val_loss = self._validate()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"fine_tuned_encoder_e{epoch+1}.pt")
            
            # Early stopping (optional)
            if self._early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
    def _train_epoch(self):
        total_loss = 0.0
        for real_images, fake_images in self.train_loader:
            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)
            
            real_features = self.encoder(real_images)
            fake_features = self.encoder(fake_images)

            loss = self.criterion(real_features, fake_features)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        total_loss = 0.0
        with torch.no_grad():
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
        self.data_path = "/Users/eric/Downloads/Transfer Learning"
        self.use_half = True
        self.image_resolution = 480
        self.use_aug = True
        self.aug_rotation = 15
        self.aug_scale = 1.5
        self.batch_size = 40960

# Usage
options = Options()  # Assuming you have an Options class or similar configuration
trainer = TrainerEncoder(options)
trainer.train(num_epochs=10)
trainer.save_model("fine_tuned_encoder.pt")