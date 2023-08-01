import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch_lr_finder import LRFinder
import torch.optim as optim

def findmaxLRWithOneCycle(ModelClass,device,train_loader):
  '''
  The 1cycle policy anneals the learning rate from an initial learning rate to some maximum 
  learning rate and then from that maximum learning rate to some minimum learning rate 
  much lower than the initial learning rate.
  :param ModelClass: Trianing model class name
  :param device: cuda or cpu
  :param train_loader: Trianing data
  '''
  model = ModelClass().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
  criterion = torch.nn.CrossEntropyLoss()
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
  lr_finder.plot() # to inspect the loss-learning rate graph
  lr_finder.reset() # to reset the model and optimizer to their initial state
  return lr_finder

cifar_mean, cifar_std = ( 0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def inv_normalize(dataset="cifar"):
  """
  #The inverse normalization should be
   x   = z*sigma + mean
       = (z + mean/sigma) * sigma
       = (z - (-mean/sigma)) / (1/sigma),
       since the normalization process is actually z = (x - mean) / sigma if you look carefully at the 
       documentation of transforms.Normalize.
  """
  if dataset=="cifar":
    global cifar_mean
    global cifar_std
    mean,std=cifar_mean, cifar_std
  inv_normalize = transforms.Normalize(
    mean= [-m/s for m, s in zip(mean, std)],
    std= [1/s for s in std]
  )
  return inv_normalize


def get_lr(optimizer):
    """
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']



def visualize_images(images,labels,classes,num_of_images=20,inverse_normalize=False,mean=0,std=1):
  """"
  It is important that we view as many images as possible before training or post prediction. We wil use
  matplotlib library to plot the images.

  If the data is normalize then we have to do inverse normalization as matplotlib accepts only valyes between 0 to 1 or 0 to 255. We call
  inverse normlization transform with mean and standars deviation to vbring the pixel value into one of the above two intervals for plotting.
  
  images,labels : list of images and correpsonding labels
  classes : list/tuple of class names of images
  num_of_images : no. of images to display  
  """
  figure = plt.figure()
  if inverse_normalize:
    inv_normalize=inv_normalize()
  for index in range(1, num_of_images + 1):
      plt.subplot(4, 10, index)
      plt.axis('off')
      if inverse_normalize:
        image= inv_normalize(images[index])
      plt.imshow(np.transpose(image,(1,2,0)))
      plt.title(classes[labels[index]],size='xx-small')

def get_summary(model, input_size):
    """
    Function to get the summary of the model architecture
    :param model: Object of model architecture class
    :param input_size: Input data shape (Channels, Height, Width)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    summary(network, input_size=input_size)



