import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import io

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_pairs = self._get_image_pairs()
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        left_path, right_path = self.image_pairs[idx]
        left_image = Image.fromarray(io.imread(left_path).astype(np.uint8))
        right_image = Image.fromarray(io.imread(right_path).astype(np.uint8))
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        return left_image, right_image

    def _get_image_pairs(self):
        image_pairs = []
        for subdir, _, _ in os.walk(self.root_dir):
            if subdir.endswith('_left'):
                right_dir = subdir.replace('_left', '_right')
                left_images = sorted([os.path.join(subdir, filename) for filename in os.listdir(subdir) if filename.endswith('.tif')])
                right_images = sorted([os.path.join(right_dir, filename) for filename in os.listdir(right_dir) if filename.endswith('.tif')])
                image_pairs.extend(list(zip(left_images, right_images)))
        
        return image_pairs