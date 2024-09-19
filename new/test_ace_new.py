import logging
import math
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset
from room_dataset import RoomDataset
import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer
from data import path_to_scenes, scenes

_logger = logging.getLogger(__name__)

class Options:
    def __init__(
            self,
            scene_path: Path,
            name: str,
            encoder_name: str = "ace_encoder_pretrained.pt",
        ):

        self.scene = scene_path

        # repository path

        with Path(__file__).parent.parent as glace_path:
            output_path = glace_path / 'output'
            if name.split('.')[-1] != 'pt': name = f'{name}.pt'
            if encoder_name.split('.')[-1] != 'pt': encoder_name = f'{encoder_name}.pt'

            self.network = output_path / name
            self.encoder_path: Path = glace_path / encoder_name

        self.scene_id = -1
        self.feat_name = 'features.npy'
        self.session = ''
        self.image_resolution = 480
        self.hypotheses = 64
        self.threshold = 10
        self.inlieralpha = 100
        self.maxpixelerror = 100
        self.render_visualization = False
        self.render_target_path = Path('renderings')
        self.render_flipped_portrait = False
        self.render_sparse_queries = False
        self.render_pose_error_threshold = 20
        self.render_map_depth_filter = 10
        self.render_camera_z_offset = 4
        self.render_frame_skip = 1

def test_network(opt):
    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset
    if scene_path.suffix == '.txt':
        testset = RoomDataset(
            scene_path, scene_id=opt.scene_id, training=False, 
            mode=0, image_height=opt.image_resolution, 
            feat_name=opt.feat_name,
        )
    else:
        testset = CamLocDataset(
            scene_path / "test", mode=0, 
            image_height=opt.image_resolution, 
            feat_name=opt.feat_name,
        )

    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader
    testset_loader = DataLoader(testset, shuffle=False, num_workers=1)

    # Load network weights
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = network.to(device)
    network.eval()

    # Setup output files
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics
    avg_batch_time = 0
    num_batches = 0
    rErrs = []
    tErrs = []
    pct100_5 = pct50 = pct20 = pct10 = 0

    # Setup visualizer if needed
    if opt.render_visualization:
        target_path = vutil.get_rendering_target_path(opt.render_target_path, opt.network)
        ace_visualizer = ACEVisualizer(
            target_path, opt.render_flipped_portrait, 
            opt.render_map_depth_filter, 
            reloc_vis_error_threshold=opt.render_pose_error_threshold,
        )
        trainset = CamLocDataset(
            scene_path / "train", mode=0, 
            image_height=opt.image_resolution, 
            feat_name=opt.feat_name,
        )
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)
        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip
        )
    else:
        ace_visualizer = None

    # Testing loop
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames, global_feat, idx in testset_loader:
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)
            global_feat = global_feat.to(device, non_blocking=True)

            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW, global_feat)

            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):
                
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                frame_name = Path(frame_path).name
                if scene_path.suffix == '.txt':
                    subscene_name = Path(frame_path).parts[-4]
                    frame_name = f"{subscene_name}_{frame_name}"

                out_pose = torch.zeros((4, 4))
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0), out_pose, 
                    opt.hypotheses, opt.threshold, focal_length, ppX, ppY, 
                    opt.inlieralpha, opt.maxpixelerror, network.OUTPUT_SUBSAMPLE
                )

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = Rotation.from_matrix(r_err).as_rotvec()
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries
                    )

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 1.0:
                    pct100_5 += 1
                if r_err < 5 and t_err < 0.5:
                    pct50 += 1
                if r_err < 2 and t_err < 0.2:
                    pct20 += 1
                if r_err < 1 and t_err < 0.1:
                    pct10 += 1

                out_pose = out_pose.inverse()
                t = out_pose[0:3, 3]
                R = out_pose[0:3, 0:3].numpy()
                rot = Rotation.from_matrix(R).as_rotvec()
                angle = np.linalg.norm(rot)
                axis = rot / angle
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    avg_time = avg_batch_time / num_batches

    pct100_5 = pct100_5 / total_frames * 100
    pct50 = pct50 / total_frames * 100
    pct20 = pct20 / total_frames * 100
    pct10 = pct10 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")
    _logger.info('Accuracy:')
    _logger.info(f'\t100cm/5deg: {pct100_5:.1f}%')
    _logger.info(f'\t50cm/5deg: {pct50:.1f}%')
    _logger.info(f'\t20cm/2deg: {pct20:.1f}%')
    _logger.info(f'\t10cm/1deg: {pct10:.1f}%')
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    for scene_name, scene in scenes.items():
        print(f"Scene name: {scene_name}")
            
        for output_name, config in scene.items():
            print(f"Name {output_name}, config {config}")

            options = Options(
                scene_path = path_to_scenes / output_name,
                name = output_name,
                encoder_name = config['encoder_name'],
            )

            test_network(options)