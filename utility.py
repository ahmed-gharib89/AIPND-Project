#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Ahmed Gharib
# DATE CREATED: 01/12/2020
# REVISED DATE:
# PURPOSE: Create functions that help in getting the input argument from the user
#          and preproccess and load the data for the model
#
##
# Imports python modules
import argparse
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt                        
import seaborn as sns
sns.set_style('white')

def get_train_args():
    """
    Retrieves and parses the 8 command line arguments provided by the user when
    they run the train.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 8 command line arguments. If 
    the user fails to provide some or all of the 8 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
        1. Image Folder as --dir 
        2. CNN Model Architecture as --arch with default value 'vgg'
        3. Directory to save model's checkpoint as --save_dir with default value '.' to use the same directory
        4. A float number between 0 and 1 as --learning_rate with default value 0.01
        5. A list of hidden units to use in model classifier block as --hidden units
        6. An integer number as --epochs with default value 10
        7. A boolean to switch the use of gpu for training as --gpu with default value False
        8. A boolean to plot the model summary or not
    This function returns these arguments as an ArgumentParser object.
    Param: None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() - data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
        description='Trains a new network on a dataset and save the model as a checkpoint.')
    # Create 8 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir',
                        type=str,
                        help='path to the folder of the data')

    parser.add_argument('--arch',
                        type=str,
                        default='vgg',
                        choices=['vgg', 'resnet', 'alexnet'],
                        help='name of CNN archticture to use')

    parser.add_argument('--save_dir',
                        dest='save_dir',
                        type=str,
                        default='.',
                        help="directory to save model's checkpoints")

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate value for the optimizer')

    parser.add_argument('--hidden_units',
                        action='append',
                        type=int,
                        default=[],
                        help="List of hidden units for the model's classifier block")

    parser.add_argument('--epochs',
                        dest='epochs',
                        type=int,
                        default=10,
                        help='Number of epochs for training')

    parser.add_argument('--gpu',
                        action="store_true",
                        default=False,
                        help='Use GPU for training')

    parser.add_argument('--plot_summary',
                        action="store_true",
                        default=False,
                        help='Plot the model summary')

    # Return pasrer object containing the argumnts
    return parser.parse_args()


def get_dataloaders(data_dir, batch_size=64):
    """
    Create data loaders for train, validation and test image folders for model training, validation and testing
    param: data_dir (str): Folder contains image folders for train, valid and test images
    param: batch_size (int): number of images to use as batch size for train, valid and test loaders
    returns:
        dataloaders: a dictionary that contains the dataloaders object for train, valid and test
        image_datasets: a dictionary thet contains ImageFolder objects for train, valid and test
    """
    # Setup the normalizer
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    # Transforming the data and applying image augmentation for training dataset
    train_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize the image to 224x224
                                          # Randomly rotate the image in the range of 30 degree
                                          transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                                          transforms.ToTensor(),  # Convert the numpy array that contains the image into a tensor
                                          normalizer])  # Apply the normalizer

    # Transformation for test and validation datasets
    val_test_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize the image to 224x224
                                             transforms.ToTensor(),  # Convert the numpy array that contains the image into a tensor
                                             normalizer])  # Apply the normalizer

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=val_test_transform)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=val_test_transform)

    image_datasets = {'train': train_data,
                      'valid': valid_data,
                      'test': test_data}

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader}

    # print out some data stats
    print(f'Number of training images: {len(train_data)}')
    print(f'Number of validation images: {len(valid_data)}')
    print(f'Number of test images: {len(test_data)}')

    return dataloaders, image_datasets


def plot_model_summary(model_summary):
    """
    Plot the training summary for the model
    param: model_summary: A pandas data frame with the training data 
    returns: None simply plots the data
    """
    _, ax = plt.subplots(figsize=(20, 6), ncols=2)
    ax[0].plot(model_summary.train_losses, color='#40e580')
    ax[0].plot(model_summary.valid_losses, color='#00334e')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    #ax[0].spines['left'].set_visible(False)
    ax[0].set_title('Training VS Validation Loss', fontdict={'fontsize': 20, 'fontweight':'bold'})
    ax[0].set_xlabel('Epoch', fontdict={'fontsize': 14})
    ax[0].set_ylabel('Loss', fontdict={'fontsize': 14})
    ax[0].set_ylim(0)
    ax[0].legend(['Training', 'Validation'])
    ax[1].plot(model_summary.train_acc, color='#40e580')
    ax[1].plot(model_summary.valid_acc, color='#00334e')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    #ax[1].spines['left'].set_visible(False)
    ax[1].set_title('Training VS Validation Accuracy', fontdict={'fontsize': 20, 'fontweight':'bold'})
    ax[1].set_xlabel('Epoch', fontdict={'fontsize': 14})
    ax[1].set_ylabel('Accuracy', fontdict={'fontsize': 14})
    ax[1].set_ylim(0)
    ax[1].legend(['Training', 'Validation'])
    plt.show()


def process_image(image):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    param: image (str): Path for the image to preprocess
    returns: preproccessed image 
    """
    
    # Process a PIL image for use in a PyTorch model
    # First load the image
    img = Image.open(image)
    
    # Setup the normalizer
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    # Setting up image preprocessor
    img_transform = transforms.Compose([transforms.Resize((224, 224)), # Resize the image to 244x244
                                        transforms.ToTensor(), # Convert the numpy array that contains the image into a tensor
                                        normalizer]) # Apply the normalizer
    
    # Apply the preprocessing to the image 
    preproccessed_img = img_transform(img)
    
    return preproccessed_img
