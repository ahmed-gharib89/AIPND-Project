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
from PIL import Image, ImageFile
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


def get_train_args():
    """
    Retrieves and parses the 15 command line arguments provided by the user when
    they run the train.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 15 command line arguments. If 
    the user fails to provide some or all of the 15 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
        1. Image Folder as data_dir 
        2. Image size to use as input size for the model as --image_size
        3. Batch size for data loaders as --batch_size
        4. CNN Model Architecture as --arch with default value 'vgg'
        5. Directory to save model's checkpoint as --save_dir with default value '.' to use the same directory
        6. A float number between 0 and 1 as --learning_rate with default value 0.01
        7. A float number between 0 and 1 as Drop out probability to use as --drop_p default value 0.5
        8. A list of hidden units to use in model classifier block as --hidden units
        9. An integer number as --epochs with default value 10
        10. Number of epochs for training to stop if valid loss stops decreasing as --early_stopping default None
        11. A boolean to switch the use of gpu for training as --gpu with default False
        12. A boolean to plot the model training history or not as --plot_history default False
        13. A booean to Set the model for evaluation only and prevent retraining as --evaluate_only default False.
        14. A File path to save the model training history to csv file as --save_history default None
        15  A File path to Model's checkpoint to use for retraining as --checkpoint defauly None
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

    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Image size to use as input size for the model')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for data loaders')

    parser.add_argument('--arch',
                        type=str,
                        default='vgg',
                        choices=['vgg', 'resnet', 'alexnet', 'densnet', 'googlenet', 'inception'],
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

    parser.add_argument('--drop_p',
                        type=float,
                        default=0.5,
                        help='Drop out probability to use')

    parser.add_argument('--hidden_units',
                        nargs='+',
                        type=int,
                        default=[512],
                        help="List of hidden units for the model's classifier block")

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs for training')

    parser.add_argument('--early_stopping',
                        type=int,
                        default=None,
                        help='Number of epochs for training to stop if valid loss stops decreasing')

    parser.add_argument('--gpu',
                        action="store_true",
                        default=False,
                        help='Use GPU for training')

    parser.add_argument('--plot_history',
                        action="store_true",
                        default=False,
                        help='Plot the model training history')

    parser.add_argument('--evaluate_only',
                        action="store_true",
                        default=False,
                        help='Sets the model for evaluation only and prevent retraining')

    parser.add_argument('--save_history',
                        type=str,
                        default=None,
                        help='Saves the model training history to csv file')

    parser.add_argument('--check_point',
                        type=str,
                        default=None,
                        help="Model's checkpoint to use for retraining")
    
    # Return pasrer object containing the argumnts
    return parser.parse_args()

def get_predict_args():
    """
    Retrieves and parses the 6 command line arguments provided by the user when
    they run the predict.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 6 command line arguments. If 
    the user fails to provide some or all of the 6 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
        1. Image path to predict as input 
        2. A File path to Model's checkpoint to use for retraining as checkpoint 
        3. Number of top classes for predictions as --top_k default 5
        4. A file path to category names json file as --category_names default None
        5. A boolean to switch the use of gpu for prediction as --gpu with default False
        6. A boolean to plot the prediction image a long with top k classes or not as --plot_predictions default False
    This function returns these arguments as an ArgumentParser object.
    Param: None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() - data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(
        description='Use a pretrained model checkpoint for prediction.')
    # Create 8 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('input',
                        type=str,
                        help='path to the folder of the data')
    
    parser.add_argument('check_point',
                        type=str,
                        help="Model's checkpoint to use for prediction")
    
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Number of top classes for predictions')
    
    parser.add_argument('--category_names',
                        type=str,
                        default=None,
                        help='FIle path to category names json file')

    parser.add_argument('--gpu',
                        action="store_true",
                        default=False,
                        help='Use GPU for prediction')

    parser.add_argument('--plot_predictions',
                        action="store_true",
                        default=False,
                        help='Plot the model training history')
    
    # Return pasrer object containing the argumnts
    return parser.parse_args()


def get_dataloaders(data_dir, image_size=224, batch_size=64):
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
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)),  # Resize the image to 224x224
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
    train_data = datasets.ImageFolder(
        data_dir + '/train', transform=train_transform)
    valid_data = datasets.ImageFolder(
        data_dir + '/valid', transform=val_test_transform)
    test_data = datasets.ImageFolder(
        data_dir + '/test', transform=val_test_transform)

    image_datasets = {'train': train_data,
                      'valid': valid_data,
                      'test': test_data}

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader}

    # Get the number of classes
    n_classes = len(train_data.classes)

    # Get training classes weights
    classes_count = np.array([train_data.targets.count(i)
                              for i in range(n_classes)])
    weights = torch.FloatTensor(1/classes_count)

    # print out some data stats
    print(f'Number of training images: {len(train_data)}')
    print(f'Number of validation images: {len(valid_data)}')
    print(f'Number of test images: {len(test_data)}')
    print()

    return dataloaders, image_datasets, n_classes, weights


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
    img_transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize the image to 244x244
                                        transforms.ToTensor(),  # Convert the numpy array that contains the image into a tensor
                                        normalizer])  # Apply the normalizer

    # Apply the preprocessing to the image
    preproccessed_img = img_transform(img)

    return preproccessed_img
