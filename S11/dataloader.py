from torchvision import datasets,transforms

class CustomCIFAR10Dataset(datasets.CIFAR10):
  def __init__(self, root="~/data", train=True, download=True, transform=None):
    self.data=super().__init__(root=root, train=train, download=download, transform=transform)
    self.mean, self.std = ( 0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
