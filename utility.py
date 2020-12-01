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


def get_train_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the train.py program from a terminal window. This function uses Python's 
    argparse module to created and defined these 7 command line arguments. If 
    the user fails to provide some or all of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
        1. Image Folder as --dir 
        2. CNN Model Architecture as --arch with default value 'vgg'
        3. Directory to save model's checkpoint as --save_dir with default value '.' to use the same directory
        4. A float number between 0 and 1 as --learning_rate with default value 0.01
        5. A list of hidden units to use in model classifier block as --hidden units
        6. An integer number as --epochs with default value 10
        7. A boolean to switch the use of gpu for training as --gpu with default value False
    This function returns these arguments as an ArgumentParser object.
    Parameters:
        None - simply using argparse module to create & store command line arguments
    Returns:
        parse_args() - data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Trains a new network on a dataset and save the model as a checkpoint.')
    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir',
                        type=str,
                        help='path to the folder of the data')

    parser.add_argument('--arch',
                        dest='arch'
                        type=str,
                        default='vgg',
                        choices=['vgg', 'resnet', 'alexnet']
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
                        action='append'
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
    
    # Return pasrer object containing the argumnts
    return parser.parse_args()
