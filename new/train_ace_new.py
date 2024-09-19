import logging
from pathlib import Path
from typing import Optional

from ace_trainer import TrainerACE
from data import path_to_scenes, scenes

class Options:
    def __init__(
            self,
            scene_path: Path,
            name: str,
            encoder_name: str = "ace_encoder_pretrained.pt",

            checkpoint_interval: int = 5000,

            training_buffer_size: int = 4000000,
            max_iterations: int = 30000,
            mode: int = 0,
            sparse: bool = False,

            num_head_blocks: int = 1,
            num_decoder_clusters: int = 1,
            mlp_ratio: float = 1.0,
        ):
        self.scene: Path = scene_path

        with Path(__file__).parent.parent as glace_path:
            output_path = glace_path / 'output'
            if name.split('.')[-1] != 'pt': name = f'{name}.pt'

            self.output_map_file: Path = output_path / name
            self.checkpoint_path: Path = output_path / 'checkpoint' / name

            self.encoder_path: Path = glace_path / encoder_name

        self.checkpoint_interval = checkpoint_interval

        self.training_buffer_size = training_buffer_size
        self.max_iterations: max_iterations
        self.mode = mode
        self.sparse = sparse
        self.num_head_blocks = num_head_blocks
        self.mlp_ratio = mlp_ratio
        self.num_decoder_clusters = num_decoder_clusters

        # Default options
        self.global_feat: bool = True
        self.feat_name: str = 'features.npy'
        self.feat_noise_std: float = 0.1
        self.head_channels: int = 768
        self.learning_rate_min: float = 0.0005
        self.learning_rate_max: float = 0.005
        self.samples_per_image: int = 1024
        self.batch_size: int = 40960
        self.repro_loss_hard_clamp: int = 1000
        self.repro_loss_soft_clamp: int = 50
        self.repro_loss_soft_clamp_min: int = 1
        self.use_half: bool = True
        self.use_homogeneous: bool = True
        self.use_aug: bool = True
        self.aug_rotation: int = 15
        self.aug_scale: float = 1.5
        self.image_resolution: int = 480
        self.repro_loss_type: str = "dyntanh"
            # choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"]
        self.repro_loss_schedule: str = "circle"
            # choices=["circle", "linear"]
        self.depth_min: float = 0.1
        self.depth_target: float = 10
        self.depth_max: float = 1000
        self.num_clusters: Optional[int] = None
        self.cluster_idx: Optional[int] = None
        self.render_visualization: bool = False
        self.render_target_path: Path = Path('renderings')
        self.render_flipped_portrait: bool = False
        self.render_map_error_threshold: int = 10
        self.render_map_depth_filter: int = 10
        self.render_camera_z_offset: int = 4

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Options has no attribute '{key}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    for scene_name, scene in scenes.items():
        print(f"Scene name: {scene_name}")
            
        for output_name, config in scene.items():
            print(f"Name {output_name}, config {config}")

            options = Options(
                scene_path = path_to_scenes / output_name,
                name = output_name,
                mode = config['mode'],
                sparse = config['sparse'],
                num_head_blocks = config['num_head_blocks'],
                num_decoder_clusters = config['num_decoder_clusters'],

                # other settings
                # training_buffer_size: int = 4000000,
                # max_iterations: int = 30000,
                # mlp_ratio: float = 1.0,
            )
        

            trainer = TrainerACE(options)
            trainer.train()