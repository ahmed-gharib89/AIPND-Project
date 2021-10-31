#!/bin/env python
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Ahmed Gharib
# DATE CREATED: 03/12/2020
# REVISED DATE:
# PURPOSE: Create a training script file to use from terminal
#
##
# Imports python modules
from classifier import Classifier
from utility import get_train_args, get_dataloaders
import torch


def main():
    """
    Main Function to use when file called from terminal
    """
    # Get the user inputs
    args = get_train_args()
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
    print('Training on ' + 'GPU' if device == 'cuda' else 'CPU')
    print()

    # Initialize the classifier class
    model = Classifier(device)

    # Load the data
    dataloaders, image_datasets, n_classes, weights = get_dataloaders(
        data_dir=args.data_dir, image_size=args.image_size, batch_size=args.batch_size)

    # Compile the model if no checkpoint is used or load the model from
    # checkpoint
    if args.check_point is None:
        model.compile(
            n_classes,
            arch=args.arch,
            hidden_units=args.hidden_units,
            drop_p=args.drop_p,
            learning_rate=args.learning_rate,
            weights=weights)
    else:
        model.load(args.check_point)

    # Train and save the model if not on evaluate only mode
    if not args.evaluate_only:
        model.train(
            args.epochs,
            dataloaders,
            image_datasets,
            early_stopping=args.early_stopping)
        model.save(args.save_dir)

    # Evaluate the model in test data
    model.evaluate(dataloaders)

    # Plot the model history if True
    if args.plot_history:
        model.plot_training_history()

    # Save model training history to csv file if True
    if not args.save_history is None:
        model.save_training_history(args.save_history)


if __name__ == '__main__':
    main()
