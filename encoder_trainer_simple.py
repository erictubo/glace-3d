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

# def collate_with_negatives(batch):
#     # Assume each item in batch is (anchor, positive)
#     anchors, positives = zip(*batch)
    
#     # Create a list of all positives
#     all_positives = list(positives)
    
#     # For each anchor, select a random item from all_positives as negative
#     # Ensure it's not the same as the positive for that anchor
#     negatives = []
#     for i, anchor in enumerate(anchors):
#         negative_candidates = all_positives[:i] + all_positives[i+1:]
#         negative = random.choice(negative_candidates)
#         negatives.append(negative)
    
#     # Convert to tensors
#     anchors = torch.stack(anchors)
#     positives = torch.stack(positives)
#     negatives = torch.stack(negatives)
    
#     return anchors, positives, negatives

def custom_collate_with_negatives(batch):
    """
    Custom collate function to pad images to the same size and include negative samples.
    """
    real_images = [item[0] for item in batch]
    fake_images = [item[1] for item in batch]
    
    # Create negative samples by shuffling fake images
    negative_images = fake_images.copy()
    random.shuffle(negative_images)

    # Ensure negatives are different from positives
    for i in range(len(fake_images)):
        if torch.all(torch.eq(fake_images[i], negative_images[i])):
            j = (i + 1) % len(fake_images)  # Choose the next image as negative
            negative_images[i], negative_images[j] = negative_images[j], negative_images[i]
    
    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images + fake_images + negative_images])
    max_width = max([img.shape[2] for img in real_images + fake_images + negative_images])
    
    # Pad images
    def pad_images(images):
        return [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in images]
    
    real_images_padded = pad_images(real_images)
    fake_images_padded = pad_images(fake_images)
    negative_images_padded = pad_images(negative_images)
    
    return (torch.stack(real_images_padded), 
            torch.stack(fake_images_padded), 
            torch.stack(negative_images_padded))


# class LossLogger:
#     def __init__(self, log_interval):
#         self.log_interval = log_interval
#         self.losses = {}

#     def log(self, loss_dict, iteration):
#         for key, value in loss_dict.items():
#             if key not in self.losses:
#                 self.losses[key] = 0
#             self.losses[key] += value

#         if iteration % self.log_interval == 0:
#             mean_losses = {k: v / self.log_interval for k, v in self.losses.items()}
#             log_str = f'Iter {iteration:6d}, '
#             log_str += ', '.join([f'{k}: {v:.6f}' for k, v in mean_losses.items()])
#             _logger.info(log_str)

#             self.losses = {}


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


        # Initial encoder (static) to be used for reference
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
            collate_fn=custom_collate_with_negatives,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=custom_collate_with_negatives,
        )
        
        _logger.info(f"Loaded training and validation datasets")


        # Initialize optimizer
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=options.learning_rate,
        )

        self.scaler = GradScaler(enabled=self.options.use_half)


        self.loss_logger = LossLogger(log_interval=self.options.gradient_accumulation_steps)


        self.encoder.eval()
        val_loss = self._validate()

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

        for real_images, fake_images, other_fake_images in self.train_loader:

            real_images = real_images.to(self.device)
            fake_images = fake_images.to(self.device)
            other_fake_images = other_fake_images.to(self.device)

            loss_contrastive = self.contrastive_loss(real_images, fake_images, other_fake_images, mode='training')


            loss = loss_contrastive / self.options.gradient_accumulation_steps

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
            #         f'Loss: {loss_contrastive:.6f}, '
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
    
    def _validate(self):
        
        self.loss_logger.reset_val()

        total_loss = 0.0

        with torch.no_grad(), autocast(enabled=self.options.use_half):
            for real_images, fake_images, other_fake_images in self.val_loader:

                real_images = real_images.to(self.device)
                fake_images = fake_images.to(self.device)
                other_fake_images = other_fake_images.to(self.device)

                loss = self.contrastive_loss(real_images, fake_images, other_fake_images, mode='validation')

                total_loss += loss.item()

        return total_loss / len(self.val_loader)
    
    def _cosine_loss(self, features_1, features_2, target_value=1, margin=0.1):

        assert features_1.shape == features_2.shape

        features_1 = F.normalize(features_1.view(features_1.size(0), -1), dim=1)
        features_2 = F.normalize(features_2.view(features_2.size(0), -1), dim=1)

        target = target_value * torch.ones(features_1.size(0)).to(self.device)

        loss_fn = nn.CosineEmbeddingLoss(margin=margin)
        loss = loss_fn(features_1, features_2, target)

        return loss
    
    def contrastive_loss(self, anchor, positive, negative, mode):

        assert not torch.all(torch.eq(positive, negative)), "Negative samples are the same as positive samples"

        assert mode in ['training', 'validation']

        with autocast(enabled=self.options.use_half):

            with torch.no_grad():
                anchor_initial_features = self.initial_encoder(anchor)
                # positive_initial_features = self.initial_encoder(positive)
                # negative_initial_features = self.initial_encoder(negative)

            with torch.no_grad() if mode=='validation' else torch.enable_grad():

                anchor_features = self.encoder(anchor)
                positive_features = self.encoder(positive)
                negative_features = self.encoder(negative)

            anchor_vs_positive = self._cosine_loss(anchor_features, positive_features)

            positive_vs_negative = self._cosine_loss(positive_features, negative_features, target_value=-1)

            anchor_vs_anchor_initial = self._cosine_loss(anchor_features, anchor_initial_features)

            # TODO: put weights and logging interval in options / class variables / function arguments

            a = 0.4
            b = 0.3
            c = 0.3

            contrastive_loss = a * anchor_vs_positive + b * positive_vs_negative + c * anchor_vs_anchor_initial


            # logging_interval = self.gradient_accumulation_steps

            # if logging:

            #     if self.iteration == 0:
            #         total_anchor_vs_positive = anchor_vs_positive
            #         total_positive_vs_negative = positive_vs_negative
            #         total_anchor_vs_anchor_initial = anchor_vs_anchor_initial
            #         total_contrastive_loss = contrastive_loss

            #     elif self.iteration % logging_interval != 0:
            #         total_anchor_vs_positive += anchor_vs_positive
            #         total_positive_vs_negative += positive_vs_negative
            #         total_anchor_vs_anchor_initial += anchor_vs_anchor_initial
            #         total_contrastive_loss += contrastive_loss

            #     elif self.iteration % logging_interval == 0:
                    
            #         # Calculate mean loss over logging interval
            #         mean_anchor_vs_positive = total_anchor_vs_positive / self.options.gradient_accumulation_steps
            #         mean_positive_vs_negative = total_positive_vs_negative / self.options.gradient_accumulation_steps
            #         mean_anchor_vs_anchor_initial = total_anchor_vs_anchor_initial / self.options.gradient_accumulation_steps
            #         mean_contrastive_loss = total_contrastive_loss / self.options.gradient_accumulation_steps

            #         _logger.info(
            #             f'Iter {self.iteration:6d}, '
            #             f'A-P:  {mean_anchor_vs_positive:.6f}, '
            #             f'P-N:  {mean_positive_vs_negative:.6f}, '
            #             f'A-AI: {mean_anchor_vs_anchor_initial:.6f}, '
            #             f'Contrastive: {mean_contrastive_loss:.6f}'
            #         )

            #         # Restart logging
            #         total_anchor_vs_positive = anchor_vs_positive
            #         total_positive_vs_negative = positive_vs_negative
            #         total_anchor_vs_anchor_initial = anchor_vs_anchor_initial
            #         total_contrastive_loss = contrastive_loss

            loss_dict = {
                'A-P': anchor_vs_positive.item(),
                'P-N': positive_vs_negative.item(),
                'A-AI': anchor_vs_anchor_initial.item(),
                'Contrastive': contrastive_loss.item()
            }
            
            self.loss_logger.log(loss_dict, self.iteration, mode)
            

            # # Add L2 regularization
            # l2_reg = torch.tensor(0., device=self.device)
            # for param in self.encoder.parameters():
            #     l2_reg += torch.norm(param)
            
            # lambda_reg = 0.5* 1e-3 # weight of about 0.25
            # loss_combined += lambda_reg * l2_reg

        return contrastive_loss


    # def combined_loss(self, real_images, fake_images, use_no_grad):

    #     with autocast(enabled=self.options.use_half):

    #         with torch.no_grad():
    #             real_initial_features = self.initial_encoder(real_images)
    #             fake_initial_features = self.initial_encoder(fake_images)

    #         with torch.no_grad() if use_no_grad else torch.enable_grad():

    #             real_features = self.encoder(real_images)
    #             fake_features = self.encoder(fake_images)

    #         fake_vs_real = self.cosine_loss(fake_features, real_features)
    #         real_vs_real_initial = self.cosine_loss(real_features, real_initial_features)

    #         fake_vs_real_initial = self.cosine_loss(fake_features, real_initial_features)
    #         fake_vs_fake_initial = self.cosine_loss(fake_features, fake_initial_features)
    #         fake_initial_vs_real_initial = self.cosine_loss(fake_initial_features, real_initial_features)

    #         w = self.options.loss_weight
    #         loss_combined = w * fake_vs_real + (1-w) * real_vs_real_initial

    #         # # Add L2 regularization
    #         # l2_reg = torch.tensor(0., device=self.device)
    #         # for param in self.encoder.parameters():
    #         #     l2_reg += torch.norm(param)
            
    #         # lambda_reg = 0.5* 1e-3 # weight of about 0.25
    #         # loss_combined += lambda_reg * l2_reg

    #     return loss_combined

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
            self.output_path = "output_encoder/fine-tuned_encoder_contrastive.pt"

            self.learning_rate = 0.0005
            # self.loss_weight = 0.5
            self.use_half = True
            self.image_resolution = 480
            self.use_aug = True
            self.aug_rotation = 15
            self.aug_scale = 1.5
            self.batch_size = 4
            self.gradient_accumulation_samples = 10

    logging.basicConfig(level=logging.INFO)

    options = Options()
    trainer = TrainerEncoder(options)
    trainer.train(num_epochs=2)
    trainer.save_model(options.output_path)
