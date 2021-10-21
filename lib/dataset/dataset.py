import torch
import os
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF

class Dataset:
    def __init__(self, directory = None, set = None):
        self.dir = os.path.join(directory, set)
        self.images_dir = os.path.join(self.dir, 'images')
        self.labels_dir = os.path.join(self.dir, 'labels')
        self.images = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.images)
    
    def transform(self, image, target):
        if random.random() >0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
        if random.random() >0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)
        target = TF.resize(target, [324,324])
        image = TF.normalize(image, mean = 0.5, std = 1)
        target = TF.normalize(target, mean = 0.5, std = 1)
        
        return image, target

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_dir,self.images[idx]))
        label = Image.open(os.path.join(self.labels_dir,self.images[idx]))
        image, label = self.transform(image, label)   
        return image, label

    