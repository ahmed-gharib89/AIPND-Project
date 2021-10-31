#!/bin/env python
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Ahmed Gharib
# DATE CREATED: 03/12/2020
# REVISED DATE:
# PURPOSE: Create a prediction script file to use from terminal
#
##
# Imports python modules
from classifier import Classifier
from utility import get_predict_args
import torch

def main():
    """
    Main Function to use when file called from terminal
    """
    # Get the user inputs
    args = get_predict_args()
    # Print the arguments
    print()
    print("Model training argument to be used")
    print('='*50)
    for arg in vars(args):
        print(f'{arg: <20}: {getattr(args, arg)}')
    
    print('='*50)
    print()

    # Use GPU if available 
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print('predicting using ' + 'GPU' if device == 'cuda' else 'CPU')
    print()

    # Initialize the classifier class
    model = Classifier(device)


    # Load the model from checkpoint
    model.load(args.check_point)

    # Load category names if available
    if not args.category_names is None:
        model.load_cat(args.category_names)

    # Predict the image
    model.predict(args.input, topk=args.top_k, plot_predictions=args.plot_predictions)

if __name__ == '__main__':
    main()