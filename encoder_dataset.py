import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms import functional as TF
from skimage import io
from skimage import color
from skimage.transform import rotate


# File structure:
# - train/
#   - render/rgb/*
#   - real/rgb/*
# - test/
#   - render/rgb/*
#   - real/rgb/*

class RealFakeDataset(Dataset):
    def __init__(
            self,
            root_dir,
            augment=False,
            # aug_rotation=15,
            # aug_scale_min=2 / 3,
            # aug_scale_max=3 / 2,
            # aug_black_white=0.1,
            # aug_color=0.3,
            # image_height=480,
            # use_half=True,

            use_half=True,
            image_height=480,
            aug_rotation=40,  # Increased to 40 degrees
            aug_scale_min=240/480,  # Minimum scale factor
            aug_scale_max=960/480,  # Maximum scale factor
            aug_brightness=0.4,
            aug_contrast=0.4,
            aug_saturation=0.3,
            aug_hue=0.3,
        ):

        self.use_half = use_half

        self.image_height = image_height

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_saturation = aug_saturation
        self.aug_hue = aug_hue

        # self.aug_black_white = aug_black_white
        # self.aug_color = aug_color

        root_dir = Path(root_dir)

        real_rgb_dir = root_dir / 'real' / 'rgb'
        fake_rgb_dir = root_dir / 'render' / 'rgb'

        self.real_rgb_files = sorted(real_rgb_dir.iterdir())
        self.fake_rgb_files = sorted(fake_rgb_dir.iterdir())

        assert len(self.real_rgb_files) == len(self.fake_rgb_files), \
            f'Number of real images ({len(self.real_rgb_files)}) does not match number of rendered images ({len(self.fake_rgb_files)})'
        

        # TODO: separate spatial and visual transformations
        # TODO: immplement visual transformations to real images only


        if self.augment:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(int(self.image_height * scale_factor)),
                # transforms.ColorJitter(
                #     brightness=self.aug_brightness,
                #     contrast=self.aug_contrast,
                #     saturation=self.aug_saturation,
                #     hue=self.aug_hue),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4], std=[0.25]),
            ])
        else:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(self.image_height),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4], std=[0.25]),
            ])

    def __len__(self):
        return len(self.real_rgb_files)
    
    def _load_image_pair(self, idx):
        real_image = io.imread(self.real_rgb_files[idx])
        fake_image = io.imread(self.fake_rgb_files[idx])

        if len(real_image.shape) < 3:
            real_image = color.gray2rgb(real_image)

        if len(fake_image.shape) < 3:
            fake_image = color.gray2rgb(fake_image)

        return real_image, fake_image
    
    @staticmethod
    def _resize_image(image, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image
    
    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def _get_single_item(self, idx, image_height, angle):
        try:
            real_image, fake_image = self._load_image_pair(idx)
        except Exception as e:
            logging.error(f"Error loading image pair at index {idx}: {str(e)}")
            raise

        real_image = self._resize_image(real_image, image_height)
        fake_image = self._resize_image(fake_image, image_height)

        image_mask = torch.ones((1, real_image.size[1], real_image.size[0]))

        real_image = self.image_transform(real_image)
        fake_image = self.image_transform(fake_image)

        if self.augment:            
            real_image = self._rotate_image(real_image, angle, 1, 'reflect')
            fake_image = self._rotate_image(fake_image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, 1, 'constant')
        
        if self.use_half and torch.cuda.is_available():
            real_image = real_image.half()
            fake_image = fake_image.half()
        
        image_mask = image_mask > 0
        
        assert real_image.shape == fake_image.shape, \
            f"Shape mismatch: real {real_image.shape}, fake {fake_image.shape}"
            
        return real_image, fake_image, image_mask

    def __getitem__(self, idx):
        if self.augment:
            # scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)

            # Inverse scale sampling
            scale = random.uniform(1/self.aug_scale_max, 1/self.aug_scale_min)
            scale_factor = 1/scale

        else:
            scale_factor = 1

        # Consistent scaling and rotation across batch
        image_height = int(self.image_height * scale_factor)
        angle = random.uniform(-self.aug_rotation, self.aug_rotation)

        if isinstance(idx, list):
            # Whole batch.
            tensors = [self._get_single_item(i, image_height, angle) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_height, angle)