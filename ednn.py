#!/usr/bin/env python

import torch
import torch.nn.functional as F
import h5py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

focus = [32, 32]
context = [32, 32]

def parseArgs():
    '''
    This is how we read in command line args!

    Parameters
    ----------
    None

    Returns
    -------
    A dictionary of the parsed/default arguments
    '''
    
    parser = argparse.ArgumentParser(
            prog='Go go EDNN!',
            description='This script runs an ednn with pytorch.'
        )
    
    parser.add_argument('-f', 
                        '--file', 
                        dest='file',
                        help='The h5 file where the data is read.', 
                        type=str, 
                        required=True)

    parser.add_argument('-e', 
                        '--epochs', 
                        dest='n_epochs', 
                        help='The number of epochs to run.', 
                        default=100, 
                        type=int)

    parser.add_argument('-bs', 
                        '--batchsize', 
                        dest='batch_size', 
                        help='Batch size for minibatch training.', 
                        default=32, 
                        type=int)

    parser.add_argument('-t', '--test',
                        dest='test',
                        help='If we want to test.',
                        default=False,
                        type=bool)

    parser.add_argument('-v', '--valid_pct',
                        dest='valid_pct',
                        help='If we want to test.',
                        default=0.05,
                        type=float)

    parser.add_argument('-m', '--model',
                        dest='model',
                        help='If provided, continue training or test with a saved model.',
                        default=None,
                        type=str)

    parser.add_argument('-lr', '--learning_rate',
                        dest='learning_rate',
                        help='Learning rate during training.',
                        default=1e-4,
                        type=float)

    return parser.parse_args()


def loadData(args):
    f = h5py.File(args.file)
    return f['X'], f['Y']


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main_net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU())

        self.final = torch.nn.Sequential(
            torch.nn.Linear(7200, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1),
            torch.nn.ReLU()
            )

    def forward(self, X):
        X = self.main_net(X)
        X = X.view(X.size(0), -1)
        return self.final(X)



def buildModel():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 4, stride=1, padding=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 64, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
        torch.nn.ReLU(),
        ), torch.nn.MSELoss(reduction='sum')


def padImages(images, padding='wrap'):
    if len(context) == 3:
        return np.pad(images, 
                (   (0,0),
                    (context[0], context[0]),
                    (context[1], context[1]),
                    (context[2], context[2]),
                    (0,0)
                ),
            mode=padding
            )
    elif len(context) == 2:
        return np.pad(images, 
                (   (0,0),
                    (context[0], context[0]),
                    (context[1], context[1]),
                    (0,0)
                ),
            mode=padding
            )

def splitImages(images):
    I = 256 // focus[0]
    J = 256 // focus[1]

    split_images = []
    for image in images:
        splits = []
        image = image.reshape(image.shape[:-1])
        # plt.imshow(image)
        # plt.colorbar()
        # plt.show()
        for i in range(I):
            for j in range(J):
                splits.append(image[i * focus[0] : (i + 1) * focus[0] + 2 * context[0], j * focus[1] : (j + 1) * focus[1] + 2 * context[1]])
        # for im in splits:
        #     print im.shape
        #     plt.imshow(im)
        #     plt.colorbar()
        #     plt.show()
        split_images.append(splits)
    return split_images

def trainModel(args, X, Y, model, loss_function, optimizer):
    for i in range(args.n_epochs):
        n_batches = X.shape[0] // args.batch_size + 1
        for j in range(n_batches):
            start = j*args.batch_size
            end = (j+1) * args.batch_size

            if end > X.shape[0]:
                end = X.shape[0]

            if start == end:
                continue
            
            X_tr = splitImages(padImages(X[start:end]))

            batch_preds = []
            for splits in X_tr:
                preds = 0
                for image in splits:
                    image = image.reshape((1,1,) + image.shape)
                    preds += model(torch.from_numpy(image))
                batch_preds.append(preds)
            
            Y_tr = Y[start:end]

            # preds = model(torch.from_numpy(X_tr))

            loss = loss_function(torch.stack(batch_preds), torch.from_numpy(Y_tr))

            print 'Epoch:', i, 'Batch:', j, 'Loss:', loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

def main():
    args = parseArgs()

    # read in data
    X, Y = loadData(args)
    
    # create model
    model = Net()
    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train/test 

    trainModel(args, X, Y, model, loss_function, optimizer)
    # done


if __name__ == '__main__':
    main()

