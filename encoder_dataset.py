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
    """
    Dataset class for loading corresponding real and fake image pairs,
    along with a distinct negative pair and a foreground mask for each fake image.
    """
    def __init__(
            self,
            root_dir,
            name: str,
            augment=False,
            augment_color=False,
            use_half=True,
            image_height=480,

            # TODO: add option for foreground masking
            # ignore_background=True,

            # TODO: add option for GT scene coordinates
            # use_coords=True,

            aug_rotation=40,
            aug_scale_min=240/480,
            aug_scale_max=960/480,
            
            aug_brightness=0.4,
            aug_contrast=0.4,
            aug_saturation=0.3,
            aug_hue=0.3,

            negative_sample_fraction=0.1,
        ):

        self.name = name

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
        self.intrinsics_dir = root_dir / 'calibration'
        self.fake_depth_dir = root_dir / 'depth'
        # self.real_coords_dir = root_dir / 'init'

        self.real_global_features = np.load(root_dir / 'features real.npy')
        self.fake_global_features = np.load(root_dir / 'features fake.npy')

        self.real_rgb_files = sorted(self.real_rgb_dir.iterdir())
        self.fake_rgb_files = sorted(self.fake_rgb_dir.iterdir())
        self.poses_files = sorted(self.poses_dir.iterdir())
        self.intrinsics_files = sorted(self.intrinsics_dir.iterdir())
        self.fake_depth_files = sorted(self.fake_depth_dir.iterdir())
        # self.real_coords_files = sorted(self.real_coords_dir.iterdir())


        assert len(self.real_rgb_files) == len(self.fake_rgb_files) == len(self.poses_files) == len(self.intrinsics_files) == len(self.fake_depth_files), \
            f'Number of real images ({len(self.real_rgb_files)}), rendered images ({len(self.fake_rgb_files)}), poses ({len(self.poses_files)}), \
                intrinsics ({len(self.intrinsics_files)}), and fake depth maps ({len(self.fake_depth_files)}) do not match.'

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

    def _get_negative_sample(self, idx, pose, is_landscape):
        """
        Find a sample that is farthest from the current sample in terms of distance,
        out of a random selection of samples that have the same aspect ratio.
        """

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
    
    def _convert_depth_to_coords(
            self,
            depth: np.ndarray,
            pose: np.ndarray,
            focal_length_pix: float,
            mm_to_m=True,
        ) -> np.ndarray:
        """
        Convert a depth map to 3D coordinates in the scene.
        """
        assert pose.shape == (4, 4), f"Pose shape {pose.shape} is not (4, 4)"

        if mm_to_m:
            depth = depth / 1000

        # create ndarray with xy values
        xy = np.zeros((2, depth.shape[0], depth.shape[1]))
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                xy[:, y, x] = [x, y]
        
        # set middle to zero
        xy[0] -= depth.shape[1] / 2
        xy[1] -= depth.shape[0] / 2

        # reproject
        xy[0] *= depth / focal_length_pix
        xy[1] *= depth / focal_length_pix

        eye = np.ndarray((4, depth.shape[0], depth.shape[1]))
        eye[0:2] = xy
        eye[2] = depth
        eye[3] = 1

        coords = np.matmul(pose, eye.reshape(4, -1))
        coords = coords.reshape(4, depth.shape[0], depth.shape[1])

        # set pixels with invalid depth to zero
        coords[:, depth == 0] = 0
        coords[:, depth > 1000] = 0
        
        return coords[:3].transpose(1, 2, 0) # shape (H, W, 3)
    
    def _load_image_set(self, idx):
        real_image_1 = io.imread(self.real_rgb_files[idx])
        fake_image_1 = io.imread(self.fake_rgb_files[idx])
        pose_1 = np.loadtxt(self.poses_files[idx])

        is_landscape = real_image_1.shape[1] > real_image_1.shape[0]
        negative_idx, distance = self._get_negative_sample(idx, pose_1, is_landscape)
        real_image_2 = io.imread(self.real_rgb_files[negative_idx])
        fake_image_2 = io.imread(self.fake_rgb_files[negative_idx])
        pose_2 = np.loadtxt(self.poses_files[negative_idx])

        assert negative_idx != idx, f"Negative sample {negative_idx} is the same as positive sample {idx}"

        logging.debug(f"Loaded image trio: real 1 {real_image_1.shape}, real 2 {real_image_2.shape}, fake 1 {fake_image_1.shape}, fake 2 {fake_image_2.shape}")

        for img in [real_image_1, real_image_2, fake_image_1, fake_image_2]:
            if len(img.shape) < 3:
                img = color.gray2rgb(img)

        # load depth map (npy file)
        fake_depth_1 = np.load(self.fake_depth_files[idx])
        fake_depth_2 = np.load(self.fake_depth_files[negative_idx])

        # create mask from depth map
        mask_1 = (fake_depth_1 > 0).astype(np.float32)
        mask_2 = (fake_depth_2 > 0).astype(np.float32)

        mask_1 = np.expand_dims(mask_1, axis=2)
        mask_2 = np.expand_dims(mask_2, axis=2)

        assert mask_1.shape[:2] == fake_image_1.shape[:2], f"Mask shape {mask_1.shape} does not match image shape {fake_image_1.shape}"
        assert mask_2.shape[:2] == fake_image_2.shape[:2], f"Mask shape {mask_2.shape} does not match image shape {fake_image_2.shape}"

        focal_length_1 = float(np.loadtxt(self.intrinsics_files[idx]))
        focal_length_2 = float(np.loadtxt(self.intrinsics_files[negative_idx]))

        # Fake coordinates from depth maps (same shape as image)
        fake_coords_1 = self._convert_depth_to_coords(fake_depth_1, pose_1, focal_length_1)
        fake_coords_2 = self._convert_depth_to_coords(fake_depth_2, pose_2, focal_length_2)

        # Real coordinates from initialization targets
        # real_coords_1 = torch.load(self.real_coords_files[idx])
        # real_coords_2 = torch.load(self.real_coords_files[negative_idx])
        # print("Real coords shapes: ", real_coords_1.shape, real_coords_2.shape)
        # TODO: how to handle subsampling? In dataset or in training loop?
        
        assert fake_coords_1.shape == fake_image_1.shape, f"Fake coords shape {fake_coords_1.shape} does not match image shape {fake_image_1.shape}"
        assert fake_coords_2.shape == fake_image_2.shape, f"Fake coords shape {fake_coords_2.shape} does not match image shape {fake_image_2.shape}"

        fake_glob_1 = torch.from_numpy(self.fake_global_features[idx]).float()
        fake_glob_2 = torch.from_numpy(self.fake_global_features[negative_idx]).float()
        
        # TODO: [IDEA] return as dictionary instead of tuple

        return real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx, negative_idx
        
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
            
            cropped_images.append(TF.crop(img, top, left, min_height, min_width))
        
        return cropped_images
    
    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = rotate(image, angle, order=order, mode=mode)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def _get_single_item(self, idx, image_height, angle=None):
        try:
            real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx_1, idx_2 = self._load_image_set(idx)
        except Exception as e:
            logging.error(f"Error loading image trio at index {idx}: {str(e)}")
            raise

        real_image_1 = self._resize_image(real_image_1, image_height)
        real_image_2 = self._resize_image(real_image_2, image_height)
        fake_image_1 = self._resize_image(fake_image_1, image_height)
        fake_image_2 = self._resize_image(fake_image_2, image_height)
        mask_1 = self._resize_image(mask_1, image_height)
        mask_2 = self._resize_image(mask_2, image_height)

        fake_coords_1 = torch.from_numpy(fake_coords_1).float().permute(2, 0, 1)
        fake_coords_2 = torch.from_numpy(fake_coords_2).float().permute(2, 0, 1)

        fake_coords_1 = F.interpolate(fake_coords_1.unsqueeze(0), size=(image_height, real_image_1.size[0])).squeeze(0)
        fake_coords_2 = F.interpolate(fake_coords_2.unsqueeze(0), size=(image_height, real_image_2.size[0])).squeeze(0)

        real_image_1, real_image_2, fake_image_1, fake_image_2 = \
            self._crop_to_smallest([real_image_1, real_image_2, fake_image_1, fake_image_2])

        mask_1 = TF.center_crop(mask_1, (real_image_1.size[1], real_image_1.size[0]))
        mask_2 = TF.center_crop(mask_2, (real_image_1.size[1], real_image_1.size[0]))

        fake_coords_1 = TF.center_crop(fake_coords_1, (real_image_1.size[1], real_image_1.size[0]))
        fake_coords_2 = TF.center_crop(fake_coords_2, (real_image_1.size[1], real_image_1.size[0]))

        real_image_1 = self.image_transform_normalized(real_image_1)
        fake_image_1 = self.image_transform(fake_image_1)
        real_image_2 = self.image_transform(real_image_2)
        fake_image_2 = self.image_transform(fake_image_2)

        mask_1 = torch.from_numpy(np.array(mask_1)).float().unsqueeze(0)
        mask_2 = torch.from_numpy(np.array(mask_2)).float().unsqueeze(0)

        # # Don't mask background
        # mask_1 = torch.ones((1, real_image_1.size[1], real_image_1.size[0]))
        # mask_2 = torch.ones((1, real_image_1.size[1], real_image_1.size[0]))

        if self.augment:
            if angle is None:
                angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            real_image_1 = self._rotate_image(real_image_1, angle, 1, 'reflect')
            real_image_2 = self._rotate_image(real_image_2, angle, 1, 'reflect')
            fake_image_1 = self._rotate_image(fake_image_1, angle, 1, 'reflect')
            fake_image_2 = self._rotate_image(fake_image_2, angle, 1, 'reflect')
            mask_1 = self._rotate_image(mask_1, angle, 1, 'reflect')
            mask_2 = self._rotate_image(mask_2, angle, 1, 'reflect')
            fake_coords_1 = self._rotate_image(fake_coords_1, angle, 1, 'reflect')
            fake_coords_2 = self._rotate_image(fake_coords_2, angle, 1, 'reflect')
        
        if self.use_half and torch.cuda.is_available():
            # real_image_1 = real_image_1.half()
            # real_image_2 = real_image_2.half()
            # fake_image_1 = fake_image_1.half()
            # fake_image_2 = fake_image_2.half()
            # mask_1 = mask_1.half()
            # mask_2 = mask_2.half()
            # fake_coords_1 = fake_coords_1.half()
            # fake_coords_2 = fake_coords_2.half()

            pass
        
        mask_1 = mask_1 >= 0.25
        mask_2 = mask_2 >= 0.25
        
        assert real_image_1.shape == real_image_2.shape == fake_image_1.shape == fake_image_2.shape == mask_1.shape == mask_2.shape, \
            f"Shape mismatch: real 1 {real_image_1.shape}, real 2 {real_image_2.shape}," \
            f"fake 1 {fake_image_1.shape}, fake 2 {fake_image_2.shape}," \
            f"mask 1 {mask_1.shape}, mask 2 {mask_2.shape}"
        
        assert fake_coords_1.shape == fake_coords_2.shape, f"Shape mismatch: fake coords 1 {fake_coords_1.shape}, fake coords 2 {fake_coords_2.shape}"

        assert fake_coords_1.shape[1:] == fake_image_1.shape[1:], f"Shape mismatch: fake coords 1 {fake_coords_1.shape}, fake image 1 {fake_image_1.shape}"
        
        return real_image_1, real_image_2, fake_image_1, fake_image_2, mask_1, mask_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx_1, idx_2, self.name

    def __getitem__(self, idx):
        if self.augment:
            scale = random.uniform(1/self.aug_scale_max, 1/self.aug_scale_min)
            scale_factor = 1/scale
        else:
            scale_factor = 1

        image_height = int(self.image_height * scale_factor) # same height across batch
        # angle = random.uniform(-self.aug_rotation, self.aug_rotation) # same angle across batch

        if isinstance(idx, list):
            tensors = [self._get_single_item(i, image_height, angle=None) for i in idx]
            return tensors
        else:
            return self._get_single_item(idx, image_height, angle=None)


def custom_collate(batch):
    """
    Custom collate function for DataLoader that randomly pads images to the same size.
    """
    real_images_1, real_images_2, fake_images_1, fake_images_2, masks_1, masks_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx_1, idx_2, names = zip(*batch)

    # Find max dimensions
    max_height = max([img.shape[1] for img in real_images_1])
    max_width = max([img.shape[2] for img in real_images_1])

    # def pad_tensor(x):
    #     return F.pad(x, (0, max_width - x.shape[2], 0, max_height - x.shape[1]))

    def pad_tensor(x, l_pad, r_pad, t_pad, b_pad):
        return F.pad(x, (l_pad, r_pad, t_pad, b_pad))

    padded_images = []
    for i in range(len(real_images_1)):
        h_padding = max_width - real_images_1[i].shape[2]
        v_padding = max_height - real_images_1[i].shape[1]
        
        # Generate consistent random padding for this sample
        l_pad = random.randint(0, h_padding)
        r_pad = h_padding - l_pad
        t_pad = random.randint(0, v_padding)
        b_pad = v_padding - t_pad

        # Apply the same padding to all corresponding images and mask
        real_image_1_padded = pad_tensor(real_images_1[i], l_pad, r_pad, t_pad, b_pad)
        real_image_2_padded = pad_tensor(real_images_2[i], l_pad, r_pad, t_pad, b_pad)
        fake_image_1_padded = pad_tensor(fake_images_1[i], l_pad, r_pad, t_pad, b_pad)
        fake_image_2_padded = pad_tensor(fake_images_2[i], l_pad, r_pad, t_pad, b_pad)
        mask_1_padded = pad_tensor(masks_1[i].float(), l_pad, r_pad, t_pad, b_pad).bool()
        mask_2_padded = pad_tensor(masks_2[i].float(), l_pad, r_pad, t_pad, b_pad).bool()

        # NEW: pad coordinates
        fake_coords_1_padded = pad_tensor(fake_coords_1[i], l_pad, r_pad, t_pad, b_pad)
        fake_coords_2_padded = pad_tensor(fake_coords_2[i], l_pad, r_pad, t_pad, b_pad)

        padded_images.append((real_image_1_padded, real_image_2_padded, fake_image_1_padded, fake_image_2_padded, mask_1_padded, mask_2_padded, fake_coords_1_padded, fake_coords_2_padded))

    # Unzip the padded images
    real_images_1_padded, real_images_2_padded, fake_images_1_padded, fake_images_2_padded, masks_1_padded, masks_2_padded, fake_coords_1_padded, fake_coords_2_padded = zip(*padded_images)

    # Stack the padded images
    real_images_1_padded = torch.stack(real_images_1_padded)
    real_images_2_padded = torch.stack(real_images_2_padded)
    fake_images_1_padded = torch.stack(fake_images_1_padded)
    fake_images_2_padded = torch.stack(fake_images_2_padded)
    masks_1_padded = torch.stack(masks_1_padded)
    masks_2_padded = torch.stack(masks_2_padded)
    fake_coords_1_padded = torch.stack(fake_coords_1_padded)
    fake_coords_2_padded = torch.stack(fake_coords_2_padded)

    fake_glob_1 = torch.stack(fake_glob_1)
    fake_glob_2 = torch.stack(fake_glob_2)


    for i in range(1, len(names)):
        assert names[i] == names[0]

    name = names[0]

    assert real_images_1_padded.shape == real_images_2_padded.shape == fake_images_1_padded.shape == fake_images_2_padded.shape == masks_1_padded.shape == masks_2_padded.shape, \
        f"Shape mismatch: real 1 {real_images_1_padded.shape}, real 2 {real_images_2_padded.shape}," \
            f"fake 1 {fake_images_1_padded.shape}, fake 2 {fake_images_2_padded.shape}," \
            f"mask 1 {masks_1_padded.shape}, mask 2 {masks_2_padded.shape}"
    
    assert fake_coords_1_padded.shape[2:] == fake_images_1_padded.shape[2:], f"Shape mismatch: fake coords 1 {fake_coords_1_padded.shape}, fake image 1 {fake_images_1_padded.shape}"

    return real_images_1_padded, real_images_2_padded, fake_images_1_padded, fake_images_2_padded, masks_1_padded, masks_2_padded, fake_coords_1_padded, fake_coords_2_padded, fake_glob_1, fake_glob_2, idx_1, idx_2, name


def coords_to_colors(coords):
    # 1. Convert coords to numpy array
    coords = coords.permute(1, 2, 0).numpy()

    # 2. Mask out zero values in coords
    mask = np.all(coords == [0., 0., 0.], axis=-1)
    masked_coords = np.ma.masked_array(coords, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))

    # 3. Normalize coords to [0, 1]
    min_coords = np.floor(masked_coords.min(axis=(0, 1)))
    max_coords = np.ceil(masked_coords.max(axis=(0, 1)))
    normalized_coords = (masked_coords - min_coords) / (max_coords - min_coords)

    normalized_coords = np.where(mask[:, :, np.newaxis], 1, normalized_coords)

    return normalized_coords.astype(np.float32)


if __name__ == '__main__':

    # Testing & Visualization:

    #logging.basicConfig(level=logging.DEBUG)
    import matplotlib.pyplot as plt

    data_path = "/home/johndoe/Documents/data/Transfer Learning/"
    dataset_names = ['notre dame', 'brandenburg gate', 'pantheon']


    for dataset_name in dataset_names:
        dataset = RealFakeDataset(data_path + dataset_name, dataset_name, augment=True, augment_color=True)
        
        # sample 4 random images from the dataset
        idx = random.sample(range(len(dataset)), 4)
        batch = dataset[idx]

        fig, ax = plt.subplots(4, 8)

        for i, sample in enumerate(batch):
            real_1, real_2, fake_1, fake_2, mask_1, mask_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx_1, idx_2, name = sample

            combined_mask = torch.logical_and(mask_1, mask_2)

            ax[i, 0].imshow(real_1[0], cmap='gray')
            ax[i, 1].imshow(real_2[0], cmap='gray')
            ax[i, 2].imshow(fake_1[0], cmap='gray')
            ax[i, 3].imshow(fake_2[0], cmap='gray')
            ax[i, 4].imshow(mask_1[0], cmap='gray')
            ax[i, 5].imshow(mask_2[0], cmap='gray')
            ax[i, 6].imshow(coords_to_colors(fake_coords_1), cmap='viridis')
            ax[i, 7].imshow(coords_to_colors(fake_coords_2), cmap='viridis')

            # ax[i, 6].imshow(combined_mask[0], cmap='gray')
        
        plt.show()


        batch = custom_collate(batch)

        real_images_1, real_images_2, fake_images_1, fake_images_2, masks_1, masks_2, fake_coords_1, fake_coords_2, fake_glob_1, fake_glob_2, idx_1, idx_2, name = batch

        combined_masks = torch.logical_and(masks_1, masks_2)

        print(fake_glob_1.shape)

        fig, ax = plt.subplots(4, 8)

        for i in range(4):
            ax[i, 0].imshow(real_images_1[i][0], cmap='gray')
            ax[i, 1].imshow(real_images_2[i][0], cmap='gray')
            ax[i, 2].imshow(fake_images_1[i][0], cmap='gray')
            ax[i, 3].imshow(fake_images_2[i][0], cmap='gray')
            ax[i, 4].imshow(masks_1[i][0], cmap='gray')
            ax[i, 5].imshow(masks_2[i][0], cmap='gray')
            ax[i, 6].imshow(coords_to_colors(fake_coords_1[i]), cmap='viridis')
            ax[i, 7].imshow(coords_to_colors(fake_coords_2[i]), cmap='viridis')

            # ax[i, 6].imshow(combined_masks[i][0], cmap='gray')

        plt.show()