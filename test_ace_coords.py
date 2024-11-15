#!/usr/bin/env python3
# Contribution: New addition in GLACE-3D - coordinate evaluation script

"""
Test script to evaluate scene coordinates against ground truth.

This script evaluates predicted scene coordinates directly against available
ground truth, rather than computing poses through PnP/RANSAC as in test_ace.py.

This is an addition to the original GLACE codebase for coordinate-level evaluation.

Adapted from:
- ACE (Niantic, Inc.) - Original scene coordinate regression framework
- GLACE (Wang et al., CVPR 2024) - Global Local Accelerated Coordinate Encoding
"""

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ace_network import Regressor
from dataset import CamLocDataset
from room_dataset import RoomDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from typing import Tuple

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def visualize_scene_coordinate_map(
        coordinates: Path,
        name: str,
        path_to_output: Path = None,
        output_name: str = None,
        format: str = 'npz',
        x_range: Tuple[int, int] = None, # (min, max) for color of X coordinate
        y_range: Tuple[int, int] = None, # (min, max) for color of Y coordinate
        z_range: Tuple[int, int] = None, # (min, max) for color of Z coordinate
        ignore_limit: float = 0.05,
    ):
    """
    Visualize scene coordinates as RGB image.
    
    Args:
        coordinates: 3D coordinate tensor (3, H, W)
        name: Name for output file
        path_to_output: Output directory
        output_name: Custom output filename
        format: Output format
        x_range, y_range, z_range: Coordinate ranges for normalization
        ignore_limit: Fraction of outliers to ignore
    """
    coordinates = coordinates.permute(1, 2, 0).numpy()

    cmap = plt.get_cmap('viridis')

    mask = np.all(coordinates == [0., 0., 0.], axis=-1)
    masked_coordinates = np.ma.masked_array(coordinates, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))
    
    min_coords = np.floor(masked_coordinates.min(axis=(0, 1)))
    max_coords = np.ceil(masked_coordinates.max(axis=(0, 1)))

    # print(f'Min coordinates: {min_coords}')
    # print(f'Max coordinates: {max_coords}')

    if (x_range and y_range and z_range):
        min_coords_limit = np.array([x_range[0], y_range[0], z_range[0]])
        max_coords_limit = np.array([x_range[1], y_range[1], z_range[1]])

        # Calculate the quantity of any coordinates outside the specified ranges (any dimension)
        num_coords_outside = np.sum(
            np.any((masked_coordinates < min_coords_limit) | (masked_coordinates > max_coords_limit), axis=-1)
        )

        print(f'Number of coordinates outside the specified ranges: {num_coords_outside}')

        # if less than 5% of the coordinates are outside the specified ranges, use the specified ranges
        # remove the coordinates outside the specified ranges
        if num_coords_outside < ignore_limit * masked_coordinates.size:
            masked_coordinates = np.clip(masked_coordinates, min_coords_limit, max_coords_limit)
        else:
            raise ValueError(f'Percentage of coordinates outside the specified ranges {num_coords_outside / masked_coordinates.size} is greater than the limit {ignore_limit}')

        # assert (min_coords >= min_coords_limit).all() and (max_coords <= max_coords_limit).all(), \
        #     f'min_coords {min_coords} and max_coords {max_coords} are not within the specified ranges'

        # Normalize the coordinates such that [min, max] -> [0, 1]
        normalized_coordinates = (masked_coordinates - min_coords_limit) / (max_coords_limit - min_coords_limit)

    else:
        # Normalize the coordinates to the range [0, 1]
        normalized_coordinates = (masked_coordinates - min_coords) / (max_coords - min_coords)

    # set all masked values to white
    normalized_coordinates = np.where(mask[:, :, np.newaxis], 1, normalized_coordinates)
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use the normalized coordinates as RGB values
    im = ax.imshow(normalized_coordinates, cmap=cmap)
    
    # ax.set_title('3D Coordinate Visualization')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.axis('off')

    ax.set_xticks([])
    ax.set_yticks([])
    
    if not path_to_output:
        plt.show()
    else:
        if not output_name:
            output_name = f'{name}'
        plt.savefig(path_to_output / f'{output_name}.png', transparent=True)
    plt.close()


def compare_scene_coordinate_maps(
        gt_coordinates: torch.Tensor, # ground truth scene coordinates: (3, H, W)
        coordinates: torch.Tensor,    # predicted scene coordinates: (3, H, W)
        name: str,
        path_to_output: Path = None,
        ):
    """
    Visualize pixel-wise difference between predicted and ground truth coordinates.
    
    Args:
        gt_coordinates: Ground truth coordinates (3, H, W)
        coordinates: Predicted coordinates (3, H, W)
        name: Output filename
        path_to_output: Output directory
    
    Returns:
        float: Mean difference in meters
    """
    
    # Torch tensor (3, H, W) -> numpy array (H, W, 3)
    gt_coordinates = gt_coordinates.permute(1, 2, 0).numpy()
    gt_mask = np.all(gt_coordinates == [0., 0., 0.], axis=-1)

    coordinates = coordinates.permute(1, 2, 0).numpy()

    assert gt_coordinates.shape == coordinates.shape, f"Shapes do not match: {gt_coordinates.shape} != {coordinates.shape}"

    # Calculate the difference
    difference = np.linalg.norm(coordinates - gt_coordinates, axis=-1)
    masked_difference = np.ma.masked_array(difference, mask=gt_mask)

    cmap = plt.get_cmap('Spectral').reversed()
    cmap.set_bad(color='white')
    
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a color-coded image of the depth map
    im = ax.imshow(masked_difference, cmap=cmap)

    # Reverse the y-axis
    # ax.invert_yaxis()
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Distance', rotation=270, labelpad=15)
    
    # ax.set_title('Scene Coordinate Differences')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.axis('off')

    ax.set_xticks([])
    ax.set_yticks([])
    
    if not path_to_output:
        plt.show()
    else:
        plt.savefig(path_to_output / f'{name}.png', transparent=True)
    plt.close()

    return np.mean(masked_difference)


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test scene coordinate predictions against ground truth.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('--scene_id', type=int, default=-1,
                        help='scene index in the room dataset, -1 to use all')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--sparse', type=_strtobool, default=False,
                        help='For mode 1: load sparse init targets when True, generate from depth when False.')

    parser.add_argument('--feat_name', type=str, default='features.npy',
                        help='global feature name.')
    
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--eval_path', type=Path,
                        help='path to save the evaluation results')

    # parser.add_argument('--session', '-sid', default='',
    #                     help='custom session name appended to output files, '
    #                          'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    # session = opt.session

    # SAME UNTIL HERE AS test_ace.py

    testset = CamLocDataset(
        root_dir = scene_path / "test",
        mode = 1, # ground truth scene coordinates
        sparse = opt.sparse,
        image_height = opt.image_resolution,
        feat_name = opt.feat_name,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=1)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    # output_dir = head_network_path.parent
    # scene_name = scene_path.name

    # network_name = head_network_path.stem
    # encoder_name = encoder_path.stem
    # eval_path = scene_path / "test" / (network_name + "-" + encoder_name)

    # # Setup output files.
    dist_log_file = opt.eval_path / f'avg_distances.txt'
    test_log_file = opt.eval_path / f'mean_avg_dist.txt'
    dist_log = open(dist_log_file, 'w', 1)
    test_log = open(test_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of the average distances.
    avg_distances = []

    id = 0
    total = len(testset)

    if not opt.eval_path.exists(): opt.eval_path.mkdir()

    for dir in ['init_viz', 'init_pred', 'init_viz_pred', 'init_viz_diff']:
        path = opt.eval_path / dir
        if not path.exists(): path.mkdir()

    # Testing loop.
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, gt_scene_coords_B3HW, filenames, global_feat, idx in testset_loader:

            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)
            global_feat = global_feat.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW,global_feat)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for frame_idx, (scene_coordinates_3HW, gt_scene_coords_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_scene_coords_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                # Remove path from file name
                frame_name = Path(frame_path).name.split('.')[0]

                # Save predicted scene coordinates
                torch.save(scene_coordinates_3HW, opt.eval_path / "init_pred" / f"{frame_name}.dat")

                # Visualize scene coordinates
                visualize_scene_coordinate_map(
                    gt_scene_coords_3HW,
                    frame_name,
                    opt.eval_path / "init_viz",
                )
                visualize_scene_coordinate_map(
                    scene_coordinates_3HW,
                    frame_name,
                    opt.eval_path / "init_viz_pred",
                )

                # Compare scene coordinates with GT scene coordinates
                avg_dist = compare_scene_coordinate_maps(
                    gt_scene_coords_3HW,
                    scene_coordinates_3HW,
                    frame_name,
                    opt.eval_path / "init_viz_diff",
                )

                avg_dist = round(avg_dist, 2)

                avg_distances.append(avg_dist)

                id += 1

                print(f"{id}/{total}: {avg_dist}")

                dist_log.write(f"{id} {frame_name}: {avg_dist}\n")

            
            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    
    assert id == total

    avg_distances.sort()
    median_idx = total // 2
    median_avg_distance = avg_distances[median_idx]

    print("Median average distance: ", median_avg_distance)

    test_log.write(f"Median: {median_avg_distance}\n")

    test_log.close()
    dist_log.close()
