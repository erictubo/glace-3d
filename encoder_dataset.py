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
            
            aug_brightness=0.4,
            aug_contrast=0.4,
            aug_saturation=0.3,
            aug_hue=0.3,

            negative_sample_fraction=0.1,
        ):

        self.augment = augment
        self.augment_color = augment_color
        self.use_half = use_half
        self.image_height = image_height

        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_saturation = aug_saturation
        self.aug_hue = aug_hue

        root_dir = Path(root_dir)

        self.real_rgb_dir = root_dir / 'rgb real'
        self.fake_rgb_dir = root_dir / 'rgb fake'
        self.poses_dir = root_dir / 'poses'
        self.depth_dir = root_dir / 'depth'

        self.real_rgb_files = sorted(self.real_rgb_dir.iterdir())
        self.fake_rgb_files = sorted(self.fake_rgb_dir.iterdir())
        self.poses_files = sorted(self.poses_dir.iterdir())
        self.depth_files = sorted(self.depth_dir.iterdir())

        assert len(self.real_rgb_files) == len(self.fake_rgb_files) == len(self.poses_files) == len(self.depth_files), \
            f'Number of real images ({len(self.real_rgb_files)}), rendered images ({len(self.fake_rgb_files)}), poses ({len(self.poses_files)}), and depth maps ({len(self.depth_files)}) do not match'

        self.image_transform = self._get_image_transform(normalize=False)
        self.image_transform_normalized = self._get_image_transform(normalize=True)
        
        self.negative_sample_size = int(negative_sample_fraction * self.__len__())

        self.epoch = 0

    def _get_image_transform(self, normalize=True):

        transforms_list = [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]

        if normalize:
            transforms_list.append(
                transforms.Normalize(mean=[0.4], std=[0.25]))

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
    
    def _load_image_set(self, idx):
        real_image = io.imread(self.real_rgb_files[idx])
        fake_image = io.imread(self.fake_rgb_files[idx])
        pose = np.loadtxt(self.poses_files[idx])

        is_landscape = real_image.shape[1] > real_image.shape[0]
        negative_idx, distance = self._get_negative_sample(idx, pose, is_landscape)
        diff_image = io.imread(self.fake_rgb_files[negative_idx])

        assert negative_idx != idx, f"Negative sample {negative_idx} is the same as positive sample {idx}"

        logging.debug(f"Loaded image trio: real {real_image.shape}, fake {fake_image.shape}, diff {diff_image.shape}")

        for img in [real_image, fake_image, diff_image]:
            if len(img.shape) < 3:
                img = color.gray2rgb(img)

        # load depth map (npy file) and convert to boolean mask (where depth is not 0)
        fake_depth = np.load(self.depth_files[idx])
        diff_depth = np.load(self.depth_files[negative_idx])

        fake_mask = (fake_depth > 0).astype(np.float32)
        diff_mask = (diff_depth > 0).astype(np.float32)

        fake_mask = np.expand_dims(fake_mask, axis=2)
        diff_mask = np.expand_dims(diff_mask, axis=2)


        assert fake_mask.shape[:2] == fake_image.shape[:2], f"Mask shape {fake_mask.shape} does not match image shape {fake_image.shape}"
        assert diff_mask.shape[:2] == diff_image.shape[:2], f"Mask shape {diff_mask.shape} does not match image shape {diff_image.shape}"

        return real_image, fake_image, diff_image, fake_mask, diff_mask, distance

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
        
        return farthest_idx, max_distance

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
    
    # def _create_mask_from_background(self, image, background_color=(60, 60, 60), tolerance=2):
    #     mask = np.all(np.abs(image - background_color) > tolerance, axis=-1)
    #     return torch.from_numpy(mask).float().unsqueeze(0)

    
    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = rotate(image, angle, order=order, mode=mode)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def _get_single_item(self, idx, image_height, angle=None):
        try:
            real_image, fake_image, diff_image, fake_mask, diff_mask, distance = self._load_image_set(idx)
        except Exception as e:
            logging.error(f"Error loading image trio at index {idx}: {str(e)}")
            raise

        real_image = self._resize_image(real_image, image_height)
        fake_image = self._resize_image(fake_image, image_height)
        diff_image = self._resize_image(diff_image, image_height)
        fake_mask = self._resize_image(fake_mask, image_height)
        diff_mask = self._resize_image(diff_mask, image_height)

        real_image, fake_image, diff_image = self._crop_to_smallest([real_image, fake_image, diff_image])

        fake_mask = TF.center_crop(fake_mask, (real_image.size[1], real_image.size[0]))
        diff_mask = TF.center_crop(diff_mask, (real_image.size[1], real_image.size[0]))

        # image_mask = torch.ones((1, real_image.size[1], real_image.size[0]))

        # image_mask = self._create_mask_from_background(np.array(fake_image))

        real_image = self.image_transform_normalized(real_image)
        fake_image = self.image_transform(fake_image)
        diff_image = self.image_transform(diff_image)

        fake_mask = torch.from_numpy(np.array(fake_mask)).float().unsqueeze(0)
        diff_mask = torch.from_numpy(np.array(diff_mask)).float().unsqueeze(0)

        if self.augment:
            if angle is None:
                angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            real_image = self._rotate_image(real_image, angle, 1, 'reflect')
            fake_image = self._rotate_image(fake_image, angle, 1, 'reflect')
            diff_image = self._rotate_image(diff_image, angle, 1, 'reflect')
            fake_mask = self._rotate_image(fake_mask, angle, 1, 'constant')
            diff_mask = self._rotate_image(diff_mask, angle, 1, 'constant')
        
        if self.use_half and torch.cuda.is_available():
            real_image = real_image.half()
            fake_image = fake_image.half()
            diff_image = diff_image.half()
        
        fake_mask = fake_mask > 0.5
        diff_mask = diff_mask > 0.5
        
        assert real_image.shape == fake_image.shape == diff_image.shape == fake_mask.shape == diff_mask.shape, \
            f"Shape mismatch: real {real_image.shape}, fake {fake_image.shape}, diff {diff_image.shape}, fake mask {fake_mask.shape}, diff mask {diff_mask.shape}"
            
        return real_image, fake_image, diff_image, fake_mask, diff_mask, distance

    def __getitem__(self, idx):
        if self.augment:
            scale = random.uniform(1/self.aug_scale_max, 1/self.aug_scale_min)
            scale_factor = 1/scale
        else:
            scale_factor = 1

        image_height = int(self.image_height * scale_factor)
        # angle = random.uniform(-self.aug_rotation, self.aug_rotation)

        if isinstance(idx, list):
            tensors = [self._get_single_item(i, image_height, angle=None) for i in idx]
            return tensors
        else:
            return self._get_single_item(idx, image_height, angle=None)


def custom_collate(batch):
    real_images, fake_images, diff_images, fake_masks, diff_masks, distances = zip(*batch)

    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images])
    max_width = max([img.shape[2] for img in real_images])

    # def pad_tensor(x):
    #     return F.pad(x, (0, max_width - x.shape[2], 0, max_height - x.shape[1]))

    def pad_tensor(x, l_pad, r_pad, t_pad, b_pad):
        return F.pad(x, (l_pad, r_pad, t_pad, b_pad))

    padded_images = []
    for i in range(len(real_images)):
        h_padding = max_width - real_images[i].shape[2]
        v_padding = max_height - real_images[i].shape[1]
        
        # Generate consistent random padding for this sample
        l_pad = random.randint(0, h_padding)
        r_pad = h_padding - l_pad
        t_pad = random.randint(0, v_padding)
        b_pad = v_padding - t_pad

        # Apply the same padding to all corresponding images and mask
        real_image_padded = pad_tensor(real_images[i], l_pad, r_pad, t_pad, b_pad)
        fake_image_padded = pad_tensor(fake_images[i], l_pad, r_pad, t_pad, b_pad)
        diff_image_padded = pad_tensor(diff_images[i], l_pad, r_pad, t_pad, b_pad)
        fake_mask_padded = pad_tensor(fake_masks[i].float(), l_pad, r_pad, t_pad, b_pad).bool()
        diff_mask_padded = pad_tensor(diff_masks[i].float(), l_pad, r_pad, t_pad, b_pad).bool()

        padded_images.append((real_image_padded, fake_image_padded, diff_image_padded, fake_mask_padded, diff_mask_padded))

    # Unzip the padded images
    real_images_padded, fake_images_padded, diff_images_padded, fake_masks_padded, diff_masks_padded = zip(*padded_images)

    # Stack the padded images
    real_images_padded = torch.stack(real_images_padded)
    fake_images_padded = torch.stack(fake_images_padded)
    diff_images_padded = torch.stack(diff_images_padded)
    fake_masks_padded = torch.stack(fake_masks_padded)
    diff_masks_padded = torch.stack(diff_masks_padded)

    assert real_images_padded.shape == fake_images_padded.shape == diff_images_padded.shape == fake_masks_padded.shape == diff_masks_padded.shape, \
        f"Shape mismatch: real {real_images_padded.shape}, fake {fake_images_padded.shape}, " \
        f"diff {diff_images_padded.shape}, fake mask {fake_masks_padded.shape}, diff mask {diff_masks_padded.shape}"

    return real_images_padded, fake_images_padded, diff_images_padded, fake_masks_padded, diff_masks_padded, distances


if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)
    import matplotlib.pyplot as plt

    data_path = "/home/johndoe/Documents/data/Transfer Learning/"
    dataset_names = ['notre dame', 'brandenburg gate', 'pantheon']

    for dataset_name in dataset_names:
        dataset = RealFakeDataset(data_path + dataset_name, augment=True, augment_color=True)
        
        # sample 4 random images from the dataset
        idx = random.sample(range(len(dataset)), 4)
        batch = dataset[idx]

        fig, ax = plt.subplots(4, 5)

        for i, sample in enumerate(batch):
            real, fake, diff, fake_mask, diff_mask, distance = sample

            ax[i, 0].imshow(real[0], cmap='gray')
            ax[i, 1].imshow(fake[0], cmap='gray')
            ax[i, 2].imshow(diff[0], cmap='gray')
            ax[i, 3].imshow(fake_mask[0], cmap='gray')
            ax[i, 4].imshow(diff_mask[0], cmap='gray')
        
        plt.show()


        batch = custom_collate(batch)

        real_images, fake_images, diff_images, fake_masks, diff_masks, distances = batch

        fig, ax = plt.subplots(4, 5)

        for i in range(4):
            ax[i, 0].imshow(real_images[i][0], cmap='gray')
            ax[i, 1].imshow(fake_images[i][0], cmap='gray')
            ax[i, 2].imshow(diff_images[i][0], cmap='gray')
            ax[i, 3].imshow(fake_masks[i][0], cmap='gray')
            ax[i, 4].imshow(diff_masks[i][0], cmap='gray')

        plt.show()