import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class LaneDataset(Dataset):
    def __init__(self, dataset_dir='segatten/train/dataset', subset='train', img_size=480):
        """
        :param dataset_dir: directory containing the dataset
        :param subset: 'train', 'valid', or 'test'
        :param img_size: target image size (resized to square)
        """
        super(LaneDataset, self).__init__()
        self.img_size = img_size
        self.resize_img = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_gt = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)
        self.subset = subset

        self.data_path = os.path.join(dataset_dir, subset)
        image_dir = os.path.join(self.data_path, 'images')

        # Lấy tất cả tên file ảnh có phần mở rộng .jpg hoặc .JPG
        self.filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir)
            if f.lower().endswith('.jpg')
        ]

        self.filenames.sort()
        print(f'Loaded {subset} subset with {len(self.filenames)} images')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = os.path.join(self.data_path, 'images', filename + '.JPG')
        gt_path = os.path.join(self.data_path, 'groundtruth', filename + '.png')

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path)

        img = self.resize_img(img)
        gt = self.resize_gt(gt)

        img = T.ToTensor()(img)
        gt = torch.from_numpy(np.asarray(gt).copy())  # thêm .copy() để tránh cảnh báo

        return img, gt

