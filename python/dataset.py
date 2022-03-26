import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader, Dataset
import torch
import cv2 as cv
import numpy as np
from scipy import ndimage
import random
import torchvision
from torchvision.transforms import Compose
from skimage import io

def img_to_tensor(img):
    tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    return tensor


def to_monochrome(x):
    # x_ = x.convert('L')
    x_ = np.array(x).astype(np.float32)  # convert image to monochrome
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


ImageToTensor = torchvision.transforms.ToTensor


class RandomFlip:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            for k, v in sample.items():
                sample[k] = np.flip(v, d)

        return sample


class RandomRotate90:
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            for k, v in sample.items():
                sample[k] = np.rot90(v, factor)

        return sample


class Rescale(object):
    def __init__(self, output_size, prob=0.9):
        self.prob = prob
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if random.random() < self.prob:
            image = sample["image"]
            raw_h, raw_w = image.shape[:2]
            for k, v in sample.items():
                sample[k] = cv.resize(v, (self.output_size, self.output_size))

            img = sample["image"]
            h, w = img.shape[:2]

            if h > raw_w:
                i = random.randint(0, h - raw_h)
                j = random.randint(0, w - raw_h)
                for k, v in sample.items():
                    sample[k] = v[i:i + raw_h, j:j + raw_h]

            else:
                res_h = raw_w - h
                for k, v in sample.items():
                    sample[k] = cv.copyMakeBorder(v, res_h, 0, res_h, 0, borderType=cv.BORDER_REFLECT)

            return sample
        else:
            return sample


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, sample):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            img = sample['image']
            height, width = img.shape[0:2]
            mat = cv.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            for k, v in sample.items():
                sample[k] = cv.warpAffine(v, mat, (height, width),
                                     flags=cv.INTER_LINEAR,
                                     borderMode=cv.BORDER_REFLECT_101)

        return sample


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):

        for t in self.transforms:
            sample = t(sample)
        return sample


class BIPEDDataset(Dataset):
    def __init__(self, img_root, mode='train', crop_size=None):
        # scaleList = [
        #              # int(crop_size * 0.875),
        #              crop_size,
        #              # int(crop_size * 1.125),
        #             ]
        self.img_root = img_root
        self.mode = mode
        self.imgList = os.listdir(img_root)
        self.crop_size = crop_size
        self.transforms = DualCompose([
                # Rotate(),
                RandomFlip(),
                RandomRotate90(),
                # Rescale(scaleList[random.randint(0, len(scaleList) - 1)])
            ])

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgPath = os.path.join(self.img_root, self.imgList[idx])
        assert os.path.exists(imgPath), 'please check if the image path exists'
        file_name = os.path.basename(imgPath)

        labelPath = imgPath.replace("images", "labels")
        assert os.path.exists(labelPath), f"please check your label path:{labelPath}"
        suffix = self.imgList[idx].split('.')[-1]
        #####load data
        image = io.imread(imgPath)
        label = io.imread(labelPath)
        image_shape = [image.shape[0], image.shape[1]]
        dwm = distranfwm(label)

        sample = {}
        sample['image'] = image
        sample['gt'] = label
        #sample['edge'] = edge
        #sample['gaussian'] = gaussian
        sample = self.transform(sample)

        sample['file_name'] = file_name
        sample['image_shape'] = image_shape
        return sample

    def transform(self, sample):
        img = sample['image']
        h, w = img.shape[:2]

        if self.crop_size:
            assert (self.crop_size < h and self.crop_size < w)
            i = random.randint(2, h - self.crop_size-2)
            j = random.randint(2, w - self.crop_size-2)
            for k, v in sample.items():
                sample[k] = v[i:i + self.crop_size, j:j + self.crop_size]

        sample = self.transforms(sample)
        # print(np.unique(gt))

        # wm = sample['wm']
        # wm = np.array(gt, dtype=np.float32)
        # sample['wm'] = wm

        gt = sample['gt']
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt = np.where(gt > 0, 1, 0)
        sample['gt'] = gt

        img = sample['image']
        img = np.array(img, dtype=np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        img /= 255.
        img = img.transpose((2, 0, 1))
        sample['image'] = img

        for k, v in sample.items():
            if len(v.shape) == 2:
                sample[k] = torch.from_numpy(np.array([v])).float()
            else:
                sample[k] = torch.from_numpy(v).float()

        return sample

# class balance weight map
def balancewm(mask):
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc


def distranfwm(mask, beta=3):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask != 1)
    dwm[dwm > beta] = beta
    dwm = wc + (1.0 - dwm / beta) + 1

    return dwm


if __name__ == '__main__':

    from config import Config
    cfg = Config()
    root = r'D:\2022\3\PCB\scratchData\train\train_images'
    train_dataset = BIPEDDataset(root, crop_size=384)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)
    print(len(train_loader))

    for i, data_batch in enumerate(train_loader):
        # if i > 10:
        #     break

        img, dt = data_batch['image'], data_batch['gt']

        print(img.size(), dt.size(), f"max gt : {torch.max(dt)}")
