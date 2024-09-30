import logging
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from skimage import io
from skimage import color
from skimage.transform import rotate


class RealFakeDataset(Dataset):
    def __init__(
            self,
            root_dir,
            augment=False,
            augment_color=False,
            use_half=True,
            image_height=480,

            aug_rotation=40,
            aug_scale_min=240/480,
            aug_scale_max=960/480,
            # aug_brightness=0.4,
            # aug_contrast=0.4,
            # aug_saturation=0.3,
            # aug_hue=0.3,

            negative_sample_fraction=0.1,
        ):

        self.use_half = use_half
        self.image_height = image_height
        self.augment = augment
        self.augment_color = augment_color
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

        root_dir = Path(root_dir)

        self.real_rgb_dir = root_dir / 'rgb real'
        self.fake_rgb_dir = root_dir / 'rgb fake'
        self.poses_dir = root_dir / 'poses'

        self.real_rgb_files = sorted(self.real_rgb_dir.iterdir())
        self.fake_rgb_files = sorted(self.fake_rgb_dir.iterdir())
        self.poses_files = sorted(self.poses_dir.iterdir())

        assert len(self.real_rgb_files) == len(self.fake_rgb_files) == len(self.poses_files), \
            f'Number of real images ({len(self.real_rgb_files)}), rendered images ({len(self.fake_rgb_files)}), and poses ({len(self.poses_files)}) do not match'

        self.image_transform = self._get_image_transform()
        
        self.negative_sample_size = int(negative_sample_fraction * self.__len__())

        self.epoch = 0

    def _get_image_transform(self):

        transforms_list = [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4], std=[0.25]),
        ]

        if self.augment_color:
            transforms_list.insert(0, transforms.ColorJitter(
                brightness=self.aug_brightness,
                contrast=self.aug_contrast,
                saturation=self.aug_saturation,
                hue=self.aug_hue))

        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.real_rgb_files)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def _load_image_trio(self, idx):
        real_image = io.imread(self.real_rgb_files[idx])
        fake_image = io.imread(self.fake_rgb_files[idx])
        pose = np.loadtxt(self.poses_files[idx])

        is_landscape = real_image.shape[1] > real_image.shape[0]
        negative_idx = self._get_negative_sample(idx, pose, is_landscape)
        diff_image = io.imread(self.fake_rgb_files[negative_idx])

        assert negative_idx != idx, f"Negative sample {negative_idx} is the same as positive sample {idx}"

        logging.debug(f"Loaded image trio: real {real_image.shape}, fake {fake_image.shape}, diff {diff_image.shape}")

        for img in [real_image, fake_image, diff_image]:
            if len(img.shape) < 3:
                img = color.gray2rgb(img)

        return real_image, fake_image, diff_image

    def _get_negative_sample(self, idx, pose, is_landscape):

        random.seed(idx + self.epoch * len(self))

        max_distance = 0
        farthest_idx = None
        sample_count = 0
        attempts = 0

        while sample_count < self.negative_sample_size and attempts < 100:
            sample_idx = random.randint(0, len(self) - 1)
            attempts += 1

            if sample_idx != idx:
                other_pose = np.loadtxt(self.poses_files[sample_idx])
                other_image = io.imread(self.fake_rgb_files[sample_idx])
                other_is_landscape = other_image.shape[1] > other_image.shape[0]

                if is_landscape == other_is_landscape:
                    distance = np.linalg.norm(pose[:3, 3] - other_pose[:3, 3])
                    if distance > max_distance:
                        max_distance = distance
                        farthest_idx = sample_idx
                    sample_count += 1

        if farthest_idx is None or farthest_idx == idx:
            valid_indices = [i for i in range(len(self)) if i != idx]
            return random.choice(valid_indices)
        
        return farthest_idx

    @staticmethod
    def _resize_image(image, target_height):
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)
        image = TF.resize(TF.to_pil_image(image), (target_height, target_width))
        return image
    
    @staticmethod
    def _crop_to_smallest(images):
        min_width = min(img.size[0] for img in images)
        min_height = min(img.size[1] for img in images)
        
        cropped_images = []
        for img in images:
            left = (img.size[0] - min_width) // 2
            top = (img.size[1] - min_height) // 2
            right = left + min_width
            bottom = top + min_height
            cropped_images.append(TF.crop(img, top, left, min_height, min_width))
        
        return cropped_images
    
    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        image = image.permute(1, 2, 0).numpy()
        image = rotate(image, angle, order=order, mode=mode)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def _get_single_item(self, idx, image_height, angle):
        try:
            real_image, fake_image, diff_image = self._load_image_trio(idx)
        except Exception as e:
            logging.error(f"Error loading image trio at index {idx}: {str(e)}")
            raise

        real_image = self._resize_image(real_image, image_height)
        fake_image = self._resize_image(fake_image, image_height)
        diff_image = self._resize_image(diff_image, image_height)

        real_image, fake_image, diff_image = self._crop_to_smallest([real_image, fake_image, diff_image])

        image_mask = torch.ones((1, real_image.size[1], real_image.size[0]))

        real_image = self.image_transform(real_image)
        fake_image = self.image_transform(fake_image)
        diff_image = self.image_transform(diff_image)

        if self.augment:            
            real_image = self._rotate_image(real_image, angle, 1, 'reflect')
            fake_image = self._rotate_image(fake_image, angle, 1, 'reflect')
            diff_image = self._rotate_image(diff_image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, 1, 'constant')
        
        if self.use_half and torch.cuda.is_available():
            real_image = real_image.half()
            fake_image = fake_image.half()
            diff_image = diff_image.half()
        
        image_mask = image_mask > 0
        
        assert real_image.shape == fake_image.shape == diff_image.shape == image_mask.shape, \
            f"Shape mismatch: real {real_image.shape}, fake {fake_image.shape}, diff {diff_image.shape}, mask {image_mask.shape}"
            
        return real_image, fake_image, diff_image, image_mask

    def __getitem__(self, idx):
        if self.augment:
            scale = random.uniform(1/self.aug_scale_max, 1/self.aug_scale_min)
            scale_factor = 1/scale
        else:
            scale_factor = 1

        image_height = int(self.image_height * scale_factor)
        angle = random.uniform(-self.aug_rotation, self.aug_rotation)

        if isinstance(idx, list):
            tensors = [self._get_single_item(i, image_height, angle) for i in idx]
            return tensors
        else:
            return self._get_single_item(idx, image_height, angle)


def custom_collate(batch):
    real_images, fake_images, diff_images, masks = zip(*batch)

    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images])
    max_width = max([img.shape[2] for img in real_images])

    def pad_tensor(x):
        return F.pad(x, (0, max_width - x.shape[2], 0, max_height - x.shape[1]))

    real_images_padded = torch.stack([pad_tensor(img) for img in real_images])
    fake_images_padded = torch.stack([pad_tensor(img) for img in fake_images])
    diff_images_padded = torch.stack([pad_tensor(img) for img in diff_images])
    masks_padded = torch.stack([pad_tensor(mask.float()).bool() for mask in masks])

    assert real_images_padded.shape == fake_images_padded.shape == diff_images_padded.shape == masks_padded.shape, \
        f"Shape mismatch: real {real_images_padded.shape}, fake {fake_images_padded.shape}, " \
        f"diff {diff_images_padded.shape}, mask {masks_padded.shape}"

    return real_images_padded, fake_images_padded, diff_images_padded, masks_padded