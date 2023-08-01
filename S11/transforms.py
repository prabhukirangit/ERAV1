from torchvision import datasets,transforms
from albumentations import Compose, HorizontalFlip, Normalize, Resize,Blur
import albumentations as album
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class CustomCIFAR10Dataset(datasets.CIFAR10):
  def __init__(self, root="~/data", train=True, download=True, transform=None):
    self.data=super().__init__(root=root, train=train, download=download, transform=transform)
    self.mean, self.std = ( 0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

  def apply_transformations(self,train=True):
    if train:
      list_transforms = Compose([Normalize(self.mean,self.std),
                                  album.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                  album.HorizontalFlip(),
                                  album.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                  album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.01,rotate_limit=30, p=0.5, 
                                            border_mode = cv2.BORDER_REPLICATE),
                                  album.augmentations.dropout.coarse_dropout.CoarseDropout(max_holes = 1,
                                            max_height=8, max_width=8, min_holes = 1,
                                            min_height=8, min_width=8, fill_value=self.mean, mask_fill_value = None),
                                            ToTensorV2()
                                            ])
    else:
      # Test Phase transformations
      list_transforms = Compose([Normalize(self.mean,self.std),ToTensorV2()])

    self.data.transform = transforms.Compose([transforms.Lambda(lambda x: list_transforms(image=np.array(x))['image'])])
    





