import torch
import numpy as np
import sys, os
from torchvision import transforms
import cv2, random
from PIL import Image
from util.data_aug import RandomRotate, RandomHorizontalFlip, HSV_jitter,RandomResizeCrop

USE_CV = True


class BaseDataSet(torch.utils.data.Dataset):
    def __init__(self, root, prefix_root, transforms=transforms.ToTensor()):
        self.root = root
        self.transforms = transforms
        self.prefix_root = prefix_root

        self.imgs_path = []
        self.labels = []


        with open(root, "r") as fp:
            for line in fp:
                img_path, label = line.strip().rsplit(" ", 1)
                self.imgs_path.append(os.path.join(self.prefix_root, img_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.imgs_path)


    def __getitem__(self, item):

        img_path, label = self.imgs_path[item], self.labels[item]

        if USE_CV:
            img = cv2.imread(img_path)

        else:
            img = Image.open(img_path)
            img = img.convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        # print(img.size())

        return {"img": img, "label": torch.from_numpy(np.array(int(label)))}


def get_transforms(fineSize, loadScalar, degrees=None,phase="train"):
    if USE_CV:
        transforms_list = []
        if phase == "train":
            transforms_list.append(transforms.Lambda(lambda img: RandomHorizontalFlip(img)))
            # transforms_list.append(transforms.Lambda(lambda img: RandomResizeCrop(img, scalar=loadScalar,size=(fineSize, fineSize))))
            # transforms_list.append(transforms.Lambda(lambda img: HSV_jitter(img, min_v=0.8, max_v=1.2, which="V")))
            # if degrees:
            #     transforms_list.append(transforms.Lambda(lambda img: RandomRotate(img, degrees=degrees)))

        elif phase == "val":
            # transforms_list.append(transforms.Lambda(lambda img: RandomResizeCrop(img, scalar=1.0,size=(fineSize, fineSize))))
            pass

        transforms_list.append(transforms.Lambda(lambda img: cv2.resize(img, (fineSize, fineSize))))

        transforms_list.append(transforms.Lambda(lambda img: img[:,:, (2, 1, 0)]))
        transforms_list.append(transforms.ToTensor())
        # transforms_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    else:
        transforms_list = []
        if phase == "train":
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomResizedCrop(fineSize))
            if degrees:
                transforms_list.append(transforms.RandomRotation(degrees))
        elif phase == "val":
            transforms_list.append(transforms.RandomResizedCrop(fineSize))


        transforms_list.append(transforms.ToTensor())
        # transforms_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)



def CreatDataLoader(root, batch_size, fineSize, loadScalar, degrees, phase, prefix_root=None):

    dataset = BaseDataSet(root=root, transforms=get_transforms(fineSize, loadScalar, degrees=degrees, phase=phase), prefix_root=prefix_root)


    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)