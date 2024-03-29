"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os

import pickle
import random
import numpy as np
import csv

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model import RN, BiggerRN, CNN_MLP, StateRN
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['Small_RN', 'Large_RN', 'State_RN', 'CNN_MLP',], default='Small_RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')

args = parser.parse_args()

#Use cuda if it is available
print('CUDA available?', torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

summary_writer = SummaryWriter()

#select model based on arguments
if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
elif args.model=='Small_RN':
  model = RN(args)
elif args.model=='Large_RN':
  model = BiggerRN(args)
elif args.model=='State_RN':
  model = StateRN(args)
else:
  # smaller relational network
  model = RN(args)

#trained models will be saved in
model_dirs = './model'
bs = args.batch_size

#input for state RN is the state description of an image. Set the tensor accordingly
if args.model=='State_RN':
    input_img = torch.FloatTensor(bs, 6, 14) # 6 objects per image and 14 columns to describe each object
else:
    input_img = torch.FloatTensor(bs, 3, 75, 75) # 3 color channels-red, green, and blue / 75 x 75 pixels
input_qst = torch.FloatTensor(bs, 11) # the length of question:11 (6 colors + 2 Q types + 3 Question subtypes)
label = torch.LongTensor(bs) #integer answer denoting a one hot vector's label

if args.cuda:
    print('CUDA available?', torch.cuda.is_available())
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()


input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

# Prepare input(img/sence) and label data for a batch of training sets
def tensor_data(data, i):
    # converting data from NumPy arrays to PyTorch tensors
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    #copy data from input img, qst, and label tensors to variables input_img, input_qst, and label
    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    #create a tuple of lists from a list of tuples
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel) #relational data
    norel = cvt_data_axis(norel) #non-relational data

    #for storing training accuracy and loss for each batch
    acc_rels = []
    acc_norels = []

    l_binary = []
    l_unary = []

    # train a model using batches of size <bs> from relational and non-relational datasets simultaneously
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_rel,
                   accuracy_norel))
    
    # calculate average training accuracy across batches
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy
    return avg_acc_binary, avg_acc_unary

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel) # relational data
    norel = cvt_data_axis(norel) # non-relational data

    #for storing test accuracy and loss for each batch
    accuracy_rels = []
    accuracy_norels = []

    loss_binary = []
    loss_unary = []

    # test a model using batches of size <bs> from relational and non-relational datasets simultaneously
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    #calculate average accuracy
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(accuracy_rel, accuracy_norel))

    summary_writer.add_scalars('Accuracy/test', {
        'binary': accuracy_rel,
        'unary': accuracy_norel
    }, epoch)

    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'binary': loss_binary,
        'unary': loss_unary
    }, epoch)

    return accuracy_rel, accuracy_norel


# load the generated training and test data(images, questions, answers) from pickle file 
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr-original.pickle')
    
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)

# load the generated training and test data(images, questions, answers) from pickle file and state descriptions rom the csv files
def load_data_state():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr-original.pickle')
    train_filename = os.path.join(dirs,'train_descriptions.csv')
    test_filename = os.path.join(dirs, 'test_descriptions.csv')
    
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)

    traindf = pd.read_csv(train_filename)
    testdf = pd.read_csv(test_filename)

    
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for index, datatuple in enumerate(train_datasets):
        img, relations, norelations = datatuple[0],datatuple[1],datatuple[2]
        img = traindf.loc[traindf['img_id'] == index, :].values[:,1:]
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for index, datatuple in enumerate(test_datasets):
        img, relations, norelations = datatuple[0],datatuple[1],datatuple[2]
        img = testdf.loc[testdf['img_id'] == index, :].values[:,1:]
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))

    return (rel_train, rel_test, norel_train, norel_test)    

#load data
if args.model =='State_RN':
    rel_train, rel_test, norel_train, norel_test = load_data_state()
else:
    rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

# resume training a pretrained model
if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

#training and test loops
with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_rel',
                     'train_acc_norel', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training {args.model} model...")

    for epoch in range(1, args.epochs + 1):
        train_acc_binary, train_acc_unary = train(epoch, rel_train, norel_train)
        test_acc_binary, test_acc_unary = test(epoch, rel_test, norel_test)

        csv_writer.writerow([epoch, train_acc_binary, train_acc_unary, test_acc_binary, test_acc_unary])
        # save the model after each epoch
        model.save_model(epoch)