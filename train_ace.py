#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
# Contribution: Adapted from GLACE - added mode switching, sparse options, checkpoint support

"""
Training script for GLACE-3D scene coordinate regression.

This script trains a scene-specific regression head using the GLACE framework.
It supports both unsupervised reprojection loss and supervised 3D loss modes,
with options for transfer learning and domain adaptation.

Usage:
    python train_ace.py <scene_path> <output_map_file> [options]

Example:
    python train_ace.py datasets/Cambridge_KingsCollege output/KingsCollege.pt \
        --num_head_blocks 2 --max_iterations 30000 --mode 1
"""

import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from ace_trainer import TrainerACE


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging levels
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument('scene', type=Path,
                        help='Path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('output_map_file', type=Path,
                        help='Target file for the trained network')
    
    # Loss function options
    parser.add_argument('--mode', type=int, choices=[0, 1],
                        help='Loss mode: 0=Unsupervised reprojection loss, 1=Supervised Euclidean loss')
    
    parser.add_argument('--switch_iterations', type=int, default=10000,
                        help='Iteration to switch from mode 1 to mode 0 (supervised to unsupervised)')

    parser.add_argument('--sparse', type=_strtobool, default=False,
                        help='For mode 1: load sparse init targets when True, generate from depth when False')
    
    # Checkpoint options
    parser.add_argument('--checkpoint_interval', type=int, default=5000,
                        help='Interval to save checkpoints during training')
    
    parser.add_argument('--checkpoint_path', type=Path, default=None,
                        help='Target file to save checkpoints, default: output_path/checkpoint/output_name')
    
    # Global feature options
    parser.add_argument('--global_feat', type=_strtobool, default=True,
                        help='Use global features for enhanced localization')
    
    parser.add_argument('--feat_name', type=str, default='features.npy',
                        help='Global feature file name')
    
    parser.add_argument('--feat_noise_std', type=float, default=0.1,
                        help='Standard deviation of noise added to global features')

    # Model architecture options
    parser.add_argument('--num_decoder_clusters', type=int, default=1,
                        help='Number of decoder clusters for position decoder')
    
    parser.add_argument('--head_channels', type=int, default=768,
                        help='Number of channels in the regression head')
    
    parser.add_argument('--mlp_ratio', type=float, default=1.0,
                        help='MLP ratio for residual blocks in the head')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='File containing pre-trained encoder weights')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='Number of residual blocks in the regression head')

    # Training optimization options
    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='Lowest learning rate of 1-cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='Highest learning rate of 1-cycle scheduler')

    parser.add_argument('--training_buffer_size', type=int, default=16000000,
                        help='Number of patches in the training buffer')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='Number of patches drawn from each image when creating the buffer')

    parser.add_argument('--batch_size', type=int, default=40960,
                        help='Number of patches for each parameter update (must be multiple of 512)')
    
    parser.add_argument('--max_iterations', type=int, default=30000,
                        help='Maximum number of iterations for the training loop')

    # Loss function parameters
    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='Hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='Soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='Minimum value of the soft clamping threshold when using a schedule')

    # Data processing options
    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='Train with half precision (FP16) for memory efficiency')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='Use homogeneous coordinates (4D) instead of Euclidean (3D)')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use data augmentation during training')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='Maximum in-plane rotation angle for augmentation (degrees)')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='Maximum scale factor for augmentation')

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='Base image resolution')

    # Loss function type options
    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    # Depth regularization options
    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='Enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='Default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='Enforce maximum depth of network predictions')

    # Clustering parameters (for ensemble training used in the Cambridge experiments)
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='Split the training sequence in this number of clusters. Disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='Train on images part of this cluster. Required only if --num_clusters is set')

    # Visualization options
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='Create a video of the mapping process (slows down training)')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='Target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='Flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='Reprojection error threshold for the visualization in pixels')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='To clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='Zoom out of the scene by moving render camera backwards, in meters')

    # Parse arguments
    options = parser.parse_args()

    # Create trainer and start training
    trainer = TrainerACE(options)
    trainer.train()
