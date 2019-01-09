#!/usr/bin/env python

import torch
import h5py
import os
import argparse

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

def trainModel(args, X, Y, model, loss_function, optimizer):
    for i in range(args.n_epochs):
        n_batches = X.shape[0] // args.batch_size + 1
        for j in range(n_batches):
            start = j*args.batch_size
            end = (j+1) * args.batch_size
            if end > X.shape[0]:
                end = X.shape[0]
            
            X_tr = X[start:end].reshape(args.batch_size, 1, 256, 256)
            Y_tr = Y[start:end]

            preds = model(torch.from_numpy(X_tr))

            loss = loss_function(preds, Y_tr)

            print('Epoch:', i, 'Batch:', j, 'Loss:', loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

def main():
    args = parseArgs()

    # read in data
    X, Y = loadData(args)
    
    # create model
    model, loss_function = buildModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train/test 

    trainModel(args, X, Y, model, loss_function, optimizer)
    # done


if __name__ == '__main__':
    main()

