import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.depth = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.depth)
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        depth = self.binary_loader(self.depth[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.depth_transform(depth)
        depth = torch.div(depth.float(),255.0)
        gt = self.gt_transform(gt)
        return depth, gt

    def filter_files(self):
        assert len(self.depth) == len(self.gts)
        depth = []
        gts = []
        for depth_path, gt_path in zip(self.depth, self.gts):
            dep = Image.open(depth_path)
            gt = Image.open(gt_path)
            # if dep.size == gt.size:
            depth.append(depth_path)
            gts.append(gt_path)
        self.depth = depth
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depth_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)),transforms.ToTensor()])

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.binary_loader(self.images[self.index])
        image = self.depth_transform(image).unsqueeze(0)
        image = torch.div(image.float(),255.0)
        gt = self.binary_loader(self.gts[self.index])
        gt = self.gt_transform(gt)
        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


