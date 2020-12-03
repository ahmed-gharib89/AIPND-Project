#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Ahmed Gharib
# DATE CREATED: 01/12/2020
# REVISED DATE:
# PURPOSE: Create a Classifier class a lont with methods for compile, training,
#          Evaluation, save and load the model.
#
##
# Imports python modules
from utility import process_image
import os
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# the following import is required for training to be robust to truncated images
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set the models to use
densenet121 = models.densenet121(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
googlenet = models.googlenet(pretrained=True)
inception = models.inception_v3(pretrained=True, aux_logits=False)

models = {'inception': inception, 'googlenet': googlenet, 'densenet': densenet121, 'resnet': resnet18,
          'alexnet': alexnet, 'vgg': vgg16}


class Classifier():
    def __init__(self, device):
        """
        Initialize the classifier class with the needed parameters for it's methods
        param: model(str): model name to use
        param: device: the device to use for training, or predicting CPU or GPU
        """
        self.device = device
        # initialize tracker for minimum validation loss
        self.valid_loss_min = np.Inf
        # Initialize empty lists to track training and validation losses and accuracy for each epoch
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []
        # Initialize start epoch
        self.start_epoch = 1
        # Initialize cat to name file with None
        self.cat_to_name = None

    def _create_classifier(self, n_inputs, n_outputs, hidden_units=[512], drop_p=0.5):
        """
        Create a classifier to use for the model
        param: n_inputs (int): number of input features to use
        param: n_outputs (int): number of output features to use
        param: hidden_units (list): a list of integers to use as hidden units
        param: drop_p (float): a number between 0 and 1 to be used as the probability for dropout
        """
        classifier = nn.ModuleList()
        classifier.append(nn.Linear(n_inputs, hidden_units[0]))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(drop_p))
        if len(hidden_units) > 1:
            for (h1, h2) in zip(hidden_units[:-1], hidden_units[1:]):
                classifier.append(nn.Linear(h1, h2))
                classifier.append(nn.ReLU())
                classifier.append(nn.Dropout(drop_p))

        classifier.append(nn.Linear(hidden_units[-1], n_outputs))
        return nn.Sequential(*classifier)

    def compile(self, n_outputs, arch='vgg', hidden_units=[512], drop_p=0.5, learning_rate=0.01, weights=None):
        """
        Compile the model and prepair it for training by setting the model archticture,
        number of hidden layers to use, numer of hidden units for each layer, dropout,
        optimizer, learning rate and weights
        param: arch (str): the name of the archticture to use
        param: hidden_units (list): a list with number of hidden units to use for each hidden layer
        param: drop_p (float): a number between 0 and 1 to be used as the probability for dropout
        param: learning_rate (float): a number between 0 and 1 to be used for optimizer learning rate
        param: weights (tensot): a tensor with classes weights to be used for criterion
        """
        # Create a model
        self.model = models[arch]
        self.arch = arch

        # Freezing model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the input features for the model classifier layer
        if arch == 'vgg':
            n_inputs = self.model.classifier[0].in_features
        elif arch == 'alexnet':
            n_inputs = self.model.classifier[1].in_features
        elif arch == 'densenet':
            n_inputs = self.model.classifier.in_features
        elif arch in ['resnet', 'googlenet', 'inception']:
            n_inputs = self.model.fc.in_features
        else:
            print(
                f'{arch} is not available please chose an archticture from {models.keys()}')

        # Create a sequential model to use as a classifier
        self.classifier = self._create_classifier(
            n_inputs=n_inputs, n_outputs=n_outputs, hidden_units=hidden_units, drop_p=drop_p)

        # Replace the model's classifier with the new classifier sequential layer
        if arch in ['resnet', 'googlenet', 'inception']:
            self.model.fc = self.classifier
        else:
            self.model.classifier = self.classifier

        # Move model to GPU if available
        self.model = self.model.to(self.device)

        # Create criterion object
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))

        # Create optimizer
        if arch in ['resnet', 'googlenet', 'inception']:
            self.optimizer = optim.SGD(
                self.model.fc.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.SGD(
                self.model.classifier.parameters(), lr=learning_rate)

    def train(self, n_epochs, loaders, image_datasets, early_stopping=None):
        """
        Train the model and save the best model weights to save_dir
        param: loaders: data loaders contains train, validation and test data loaders
        param: image_datasets: a dictionary thet contains ImageFolder objects for train, valid and test
        param: early_stopping (int): a number of epochs to stop training if validation loss stop decreasing
        """

        # Start time to calculate the time for training
        print()
        print('='*50)
        print('Training ......')
        train_start = time()

        # Setting early stopping count
        early_stopping_count = 0

        end_epoch = n_epochs + self.start_epoch - 1
        for epoch in range(self.start_epoch, end_epoch + 1):
            with tqdm(total=len(image_datasets['train'])) as t_epoch_pbar:
                t_epoch_pbar.set_description(f'Train-> E({epoch}/{end_epoch})')
                # Start time for epoch
                # epoch_start = time()
                # initialize variables to monitor training and validation loss
                train_loss = 0.0
                valid_loss = 0.0
                train_correct = 0.0
                train_total = 0.0
                valid_correct = 0.0
                valid_total = 0.0
                ###################
                # train the model #
                ###################
                self.model.train()
                for batch_idx, (data, target) in enumerate(loaders['train']):
                    # move to GPU if available
                    data, target = data.to(self.device), target.to(self.device)
                    # find the loss and update the model parameters accordingly
                    # clear the gradients of all optimized variables
                    self.optimizer.zero_grad()
                    # forward pass
                    output = self.model(data)
                    # calculate the batch loss
                    loss = self.criterion(output, target)
                    # backward pass
                    loss.backward()
                    # perform a single optimization step to update model parameters
                    self.optimizer.step()
                    # update training loss
                    train_loss = train_loss + \
                        ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                    # convert output probabilities to predicted class
                    pred = output.data.max(1, keepdim=True)[1]
                    # compare predictions to true label
                    train_correct += np.sum(np.squeeze(
                        pred.eq(target.data.view_as(pred))).cpu().numpy())
                    train_total += data.size(0)
                    # Update the progress bar
                    desc = f'Train-> E({epoch}/{end_epoch}) - loss={train_loss:.4f} - Acc={train_correct/train_total:.2%}'
                    t_epoch_pbar.set_description(desc)
                    t_epoch_pbar.update(data.shape[0])
                ######################
                # validate the model #
                ######################
                self.model.eval()
            with tqdm(total=len(image_datasets['valid'])) as v_epoch_pbar:
                v_epoch_pbar.set_description(f'Valid-> E({epoch}/{end_epoch})')
                for batch_idx, (data, target) in enumerate(loaders['valid']):
                    # move to GPU if available
                    data, target = data.to(self.device), target.to(self.device)
                    # update the average validation loss
                    # forward pass
                    output = self.model(data)
                    # calculate the batch loss
                    loss = self.criterion(output, target)
                    # update average validation loss
                    valid_loss = valid_loss + \
                        ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                    # convert output probabilities to predicted class
                    pred = output.data.max(1, keepdim=True)[1]
                    # compare predictions to true label
                    valid_correct += np.sum(np.squeeze(
                        pred.eq(target.data.view_as(pred))).cpu().numpy())
                    valid_total += data.size(0)
                    # Update the progress bar
                    desc = f'Valid-> E({epoch}/{end_epoch}) - loss={valid_loss:.4f} - Acc={valid_correct/(valid_total+1e-10):.2%}'
                    v_epoch_pbar.set_description(desc)
                    v_epoch_pbar.update(data.shape[0])

            # Add train and valid loss for each epoch to the train_losses and valid_losses lists
            self.train_losses.append(train_loss.cpu().numpy())
            self.valid_losses.append(valid_loss.cpu().numpy())
            self.train_acc.append(100. * train_correct / train_total)
            self.valid_acc.append(100. * valid_correct / valid_total)
            # save the model if validation loss has decreased
            if valid_loss <= self.valid_loss_min:
                print(
                    f'Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving the model weights...')

                early_stopping_count = 0

                self.best_weights = self.model.state_dict()

                self.valid_loss_min = valid_loss
            else:
                early_stopping_count += 1

            if not early_stopping is None and early_stopping_count >= early_stopping:
                break

        self.model.load_state_dict(self.best_weights)
        self.class_to_idx = image_datasets['train'].class_to_idx

        self.start_epoch = epoch + 1
        # Save Model Summary to a pandas DF
        print('Saving the model training history ...')
        history = {
            'epoch': np.arange(1, self.start_epoch, 1),
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_acc': self.train_acc,
            'valid_acc': self.valid_acc,
        }
        self.history = pd.DataFrame(history)
        print(f'Finished Training in: {time()-train_start:.0f} Seconds')
        print('='*50)
        print()

    def save(self, save_dir='.'):
        """
        Saves the model and it's attributes
        param: save_dir: a path to save the model at
        """
        # Checkpoint file path
        now = datetime.now().strftime('%H%M%S')
        checkpoint_file = save_dir + '/' + self.arch + '_checkpoint_' + now + '.pth'
        # model state dict
        model_state = {
            'arch': self.arch,
            'state_dict': self.model.state_dict(),
            'classifier': self.classifier,
            'class_to_idx': self.class_to_idx,
            'optimizer': self.optimizer,
            'optimizer_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'criterion_dict': self.criterion.state_dict(),
            'history': self.history,
            'start_epoch': self.start_epoch,
            'valid_loss_min': self.valid_loss_min
        }
        # Save the model
        torch.save(model_state, checkpoint_file)
        print(f'Save model at {checkpoint_file}\n')

    def load(self, checkpoint_file):
        """
        Loads the model and it's attributes
        param: checkpoint_file: a file path to load the model from
        """
        # Load the model state from checkpoint file
        model_state = torch.load(checkpoint_file, map_location=self.device)
        self.arch = model_state['arch']
        self.model = models[self.arch]

        # Freezing model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the model's classifier with the new classifier sequential layer
        self.classifier = model_state['classifier']
        if self.arch == 'resnet':
            self.model.fc = self.classifier
        else:
            self.model.classifier = self.classifier

        # move model to GPU if CUDA is available
        self.model.to(self.device)

        # Load the model's state dict
        self.model.load_state_dict(model_state['state_dict'])

        # Load Class to index
        self.class_to_idx = model_state['class_to_idx']

        # Load optimizer and it's state dict
        self.optimizer = model_state['optimizer']
        self.optimizer.load_state_dict(model_state['optimizer_dict'])

        # Load criterion and it's state dict
        self.criterion = model_state['criterion']
        self.criterion.load_state_dict(model_state['criterion_dict'])

        # Load model's history
        self.history = model_state['history']

        # Load start epoch
        self.start_epoch = model_state['start_epoch']

        # Load validation loss minimum
        self.valid_loss_min = model_state['valid_loss_min']

        # Load train and validation losses and accuracy
        self.train_losses = list(self.history.train_losses)
        self.valid_losses = list(self.history.valid_losses)
        self.train_acc = list(self.history.train_acc)
        self.valid_acc = list(self.history.valid_acc)

    def save_training_history(self, file_path):
        """
        A function to save the training history of the model to csv file
        param: file_path: a path to save the file at
        """
        # Save the df to csv file
        self.history.to_csv(file_path, index=False)
        print(f'Saved training history at {file_path}\n')

    def plot_training_history(self):
        """
        Plot the training history for the model
        """
        _, ax = plt.subplots(figsize=(20, 6), ncols=2)
        ax[0].plot(self.history.train_losses, color='#40e580')
        ax[0].plot(self.history.valid_losses, color='#00334e')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        # ax[0].spines['left'].set_visible(False)
        ax[0].set_title('Training VS Validation Loss', fontdict={
                        'fontsize': 20, 'fontweight': 'bold'})
        ax[0].set_xlabel('Epoch', fontdict={'fontsize': 14})
        ax[0].set_ylabel('Loss', fontdict={'fontsize': 14})
        ax[0].set_ylim(0)
        ax[0].legend(['Training', 'Validation'])
        ax[1].plot(self.history.train_acc, color='#40e580')
        ax[1].plot(self.history.valid_acc, color='#00334e')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        # ax[1].spines['left'].set_visible(False)
        ax[1].set_title('Training VS Validation Accuracy', fontdict={
                        'fontsize': 20, 'fontweight': 'bold'})
        ax[1].set_xlabel('Epoch', fontdict={'fontsize': 14})
        ax[1].set_ylabel('Accuracy', fontdict={'fontsize': 14})
        ax[1].set_ylim(0)
        ax[1].legend(['Training', 'Validation'])
        plt.show()

    def evaluate(self, loaders):
        """
        Test the model accuracy in test data
        param: loaders: data loaders contains train, validation and test data loaders
        """
        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        self.model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU if available
            data, target = data.to(self.device), target.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update average test loss
            test_loss = test_loss + \
                ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

        self.test_loss = test_loss
        self.test_accuracy = correct / total

        print('Test Loss: {:.6f}'.format(test_loss))
        print('\nTest Accuracy: %.2f%% (%2d/%2d)\n' %
              (100. * self.test_accuracy, correct, total))

    def predict(self, image_path, topk=5, plot_predictions=False):
        """
         Predict the class (or classes) of an image using a trained deep learning model.
         param: image_path: path to the image to predict
         param: topk(int): number of top classes to predict
        """
        # Preprocess the image before passing it to the model
        preproccessed_img = process_image(image_path)
        preproccessed_img = preproccessed_img.unsqueeze(0)

        # move imag to GPU if CUDA is available
        preproccessed_img = preproccessed_img.to(self.device)

        # Use VGG16 to predict the class of the image
        self.model.eval()
        pred = F.softmax(self.model.forward(preproccessed_img), dim=1)
        prob, idx = pred.topk(topk)
        prob = np.squeeze(prob.cpu().detach().numpy())
        idx = np.squeeze(idx.cpu().detach().numpy())

        # Dictionary to map the indices to their class numbers
        idx_to_class = {self.class_to_idx[i]: i for i in self.class_to_idx}

        # Getting the class names for the top k predictions
        classes = [idx_to_class[i] for i in idx]

        if not self.cat_to_name is None:
            classes = [self.cat_to_name[c].title() for c in classes]

            if not plot_predictions:
                for p, c in zip(prob, classes):
                    print(f'Class: {c:<25} Probability: {p:.2%}')

        elif not plot_predictions:
            for p, c in zip(prob, classes):
                print(f'Class: {c:<5} Probability: {p:.2%}')

        if plot_predictions:
            self.plot_predicted_classes(image_path, prob, classes)

        return prob, classes

    def plot_predicted_classes(self, image_path, prob, classes):
        """Function to plot the predictions and the top k predicted classes.
        param: image_path: path for image to plot
        param: prob: probabilities for the topk classes
        param: classes: topk predicted classes
        """
        # Show the images and a bar chart with the predictions probabilty
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.subplots_adjust(hspace=0.2, wspace=0)

        predicted_label = classes[np.argmax(prob)]
        axes[0].imshow(Image.open(image_path))
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[0].set_title(f"prediction: {predicted_label}", fontdict={
                          'fontsize': 18, 'fontweight': 'bold'})

        axes[1].bar(classes, prob, color='#33b5e5')
        axes[1].set_title(f"Top {len(classes)} predicted classes", fontdict={
                          'fontsize': 18, 'fontweight': 'bold'})
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        prob_str = [f'{p:.2%}' for p in prob]
        for i, p in enumerate(prob):
            axes[1].text(i, p+.02, prob_str[i], ha='center')

        plt.show()

    def load_cat(self, file_path):
        """
        Loads category names file
        param: file_path: file path to load
        """
        with open(file_path, 'r') as f:
            self.cat_to_name = json.load(f)
