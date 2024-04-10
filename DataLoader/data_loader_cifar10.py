import torch
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np



'''
This file contains script to get Dataset and load it in the loader

'''


mean_list = [0.4914, 0.4822, 0.4465]
std_list  = [0.2471, 0.2435, 0.2616]




#################################### Define the data transformations ######################


class getCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
      # Initialize dataset and transform
      self.dataset = dataset
      self.transform = transform

    def __len__(self):
      return len(self.dataset)  # return length of dataset

    def __getitem__(self, idx):
      image, label = self.dataset[idx]

      #convert image to numpy array
      image = np.array(image)

      if self.transform is not None:
        image = self.transform(image=image)["image"]

      return image, label

######### Step1 Transformations Start ###########


# #################### Training transformation for Session 10 Assignment ######################
# def get_train_transforms():

#   '''

#   Train data transformations for session 10

#   '''

#   return A.Compose([

#     # Add padding before cropping 
#     A.PadIfNeeded(36,36),

#     # Randomly crop back to original dimension of data
#     A.RandomCrop(32, 32),

#     # Apply horizontal flip
#     A.HorizontalFlip(p=0.5),

#     # Apply removal of box regions from the image to introduce regularization
#     A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=[q*255 for q in mean_list], mask_fill_value = None,p=0.5),

#     # Apply normalization; albumentations applies transfomation wrt to the max pixel whose default value is 255
#     A.Normalize(mean=mean_list, std = std_list),

#     #convert to tensor
#     ToTensorV2(),
#     ])



# #################### Training transformation for Session 11 Assignment ######################
def get_train_transforms():

  '''

  Train data transformations for session 11

  '''

  return A.Compose([

    # Add padding before cropping 
    A.PadIfNeeded(40,40),

    # Randomly crop back to original dimension of data
    A.RandomCrop(32, 32),

    # Apply removal of box regions from the image to introduce regularization
    A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[q*255 for q in mean_list], mask_fill_value = None,p=0.5),

    # Apply normalization; albumentations applies transfomation wrt to the max pixel whose default value is 255
    A.Normalize(mean=mean_list, std = std_list),

    #convert to tensor
    ToTensorV2(),
    ])






def get_test_transforms():

  '''

  Test data transformations

  '''
  return A.Compose([

    # Apply normalization; albumentations applies transfomation wrt to the max pixel whose default value is 255
    A.Normalize(mean=mean_list, std = std_list),

    #convert to tensor
    ToTensorV2(),

    ])


######### Step1 Transformations End   ###########





############################# Get CIFAR dataset and pass it to loader ###########################

def get_CIFARdataset_with_loader(datasettype,kwargs):
    '''
    <datasettype> can have values as 'train' or 'test'
    This function loads the CIFAR training and testing dataset.
    The datasets are loaded and then passed on for transformation

    For more information on tranformation execute the following lines:
    get_train_transforms??   # For training data transformation
    get_test_transforms??    # For testing data transformation

    '''

    if datasettype == 'train':
        train_data = getCIFAR10(datasets.CIFAR10('../data', train=True, download=True), transform=get_train_transforms_a10())  # download and load the "training" data of CIFAR and apply test_transform
        print("Training data loaded successfully. Shape of data: ",train_data.dataset.data.shape)
        return train_data.dataset.class_to_idx, torch.utils.data.DataLoader(train_data, **kwargs)    # load train data
    elif datasettype == 'test':
        test_data = getCIFAR10(datasets.CIFAR10('../data', train=False, download=True), transform=get_test_transforms())   # download and load the "test" data of CIFAR and apply test_transform
        print("Testing data loaded successfully. Shape of data: ",test_data.dataset.data.shape)
        return test_data.dataset.class_to_idx, torch.utils.data.DataLoader(test_data, **kwargs)      # load test data
    else:
        raise ValueError('Incorrect dataset type string...pass valid name from available values')
