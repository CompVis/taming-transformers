import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        
        isFile = os.path.isfile(training_images_list_file)
        isDirectory = os.path.isdir(training_images_list_file)
        
        if isFile:
           with open(training_images_list_file, "r") as f:
               paths = f.read().splitlines()            
       
        if isDirectory:
           paths = []
           for images in os.listdir(training_images_list_file):
           
               # check if the image ends with png or jpg or jpeg
               if (images.endswith(".png") or images.endswith(".jpg")\
                   or images.endswith(".jpeg")):
                   paths.append(os.path.join(training_images_list_file, images))
                   
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        
        isFile = os.path.isfile(test_images_list_file)
        isDirectory = os.path.isdir(test_images_list_file)
        
        if isFile:
            with open(test_images_list_file, "r") as f:
                paths = f.read().splitlines()            
                
        if isDirectory:
            paths = []
            for images in os.listdir(test_images_list_file):
                
                # check if the image ends with png or jpg or jpeg
                if (images.endswith(".png") or images.endswith(".jpg")\
                    or images.endswith(".jpeg")):
                    paths.append(os.path.join(test_images_list_file, images))
        
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


