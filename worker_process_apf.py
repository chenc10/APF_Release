import torch
import numpy
from datasource import *
from model import *
import torchvision
import copy
import argparse
from datetime import datetime
import sys
import os
from apf_manager import *

parser = argparse.ArgumentParser()

parser.add_argument('--master_address', type=str, default='127.0.0.1')
parser.add_argument('--world_size', type=int, default=5)
parser.add_argument('--rank', type=int, default=0)

parser.add_argument('--model', type=str, default='CNN')       
parser.add_argument('--dataset', type=str, default='Cifar10')
parser.add_argument('--initial_lr', type=float, default=0.01)
parser.add_argument('--batch_size',  type=int, default=100)

parser.add_argument('--lr_decay_interval', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--max_epoch', type=int, default=1e10)

parser.add_argument('--sync_frequency', type=int, default=1)
parser.add_argument('--frozen_frequency', type=int, default=250)
parser.add_argument('--ema_alpha', type=float, default=0.95)
parser.add_argument('--stable_threshold', type=float, default=0.1)

args = parser.parse_args()
if args.model == 'ResNet18':
    args.lr_decay_interval=30
if args.dataset == 'ImageNet':
    args.lr_decay_interval=30

CUDA = torch.cuda.is_available()

def logging(string):
    print(str(datetime.now())+' '+str(string))
    sys.stdout.flush()

def get_train_loader():
    if args.dataset == 'Mnist':
        train_loader = Mnist(args.rank, args.batch_size).get_train_data()
    if args.dataset == 'Cifar10':
        train_loader = Cifar10(args.rank, args.batch_size).get_train_data()
    if args.dataset == 'ImageNet':
        train_loader = ImageNet(args.rank, args.batch_size).get_train_data()
    if args.dataset == 'KWS':
        train_loader = KWS(args.rank, args.batch_size).get_train_data()
    return train_loader

def get_test_loader():
    if args.dataset == 'Mnist':
        test_loader = Mnist(args.rank).get_test_data()
    if args.dataset == 'Cifar10':
        test_loader = Cifar10(args.rank).get_test_data()
    if args.dataset == 'ImageNet':
        test_loader = ImageNet(args.rank).get_test_data()
    if args.dataset == 'KWS':
        test_loader = KWS(args.rank).get_test_data()
    return test_loader

def load_model():
    if args.model == 'CNN' and args.dataset == 'Mnist':
        model = CNNMnist()
    if args.model == 'CNN' and args.dataset == 'Cifar10':
        model = CNNCifar()
    if args.model == 'ResNet18' and args.dataset == 'Cifar10':
        model = ResNet18()
    if args.model == 'VGG':
        model = VGG16()
    if args.model == 'AlexNet' and args.dataset == 'ImageNet':
        model = torchvision.models.alexnet()
    if args.model == 'DenseNet' and args.dataset == 'ImageNet':
        model = torchvision.models.densenet121()
    if args.model == 'CNNKws' and args.dataset == 'KWS':
        model = CNNKws()
    if CUDA:
        torch.cuda.set_device(args.rank)
        model = model.cuda()
    if False and os.path.exists('autoencoder'+str(args.rank)+'.t7'):
        logging('===> Try resume from checkpoint')
        checkpoint = torch.load('autoencoder'+str(args.rank)+'.t7')
        model.load_state_dict(checkpoint['state'])
        logging('model loaded')
    else:
        logging('model created')
    return model

def save_model(model, epoch_id):
    logging('===> Saving models...')
    state = {
        'state': model.state_dict(),
        }
    torch.save(state, 'autoencoder-' + args.dataset + '-' + args.model + '-' + str(epoch_id) + '.t7')

def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0.0
    total_test_number = 0.0
    with torch.no_grad():
        for step, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.cuda() if CUDA else test_x
            test_y = test_y.cuda() if CUDA else test_y
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            valid_number = (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
            batch_number = len(test_y) 
            positive_test_number += valid_number
            total_test_number += batch_number
            if args.dataset == 'ImageNet':
                logging('\t test step: ' + str(step) + '; accuracy: ' + str(float(valid_number)/batch_number))
    accuracy = positive_test_number / total_test_number
    return accuracy

def run():
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    model = load_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.initial_lr*args.world_size, weight_decay=args.weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()

    apf_manager = APF_Manager(model, args.master_address, args.world_size, args.rank, args.sync_frequency, args.frozen_frequency, args.ema_alpha, args.stable_threshold)

    iter_id = 0
    epoch_id = 0
    logging('initial model parameters:\n'+str(list(model.parameters())[0][0][0])+'\n\n ----- start training -----\n')

    while epoch_id < args.max_epoch:            
        save_model(model, epoch_id)

        if epoch_id == 0 and not os.path.exists('autoencoder'+str(args.rank)+'.t7'):
            save_model(model, epoch_id)
            logging('\t## Model Saved')
        save_model(model, epoch_id)
        
        for step, (b_x, b_y) in enumerate(train_loader):
            iter_id += 1
            b_x = b_x.cuda() if CUDA else b_x
            b_y = b_y.cuda() if CUDA else b_y
            optimizer.zero_grad()
            output = model(b_x)

            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()
            apf_manager.sync(iter_id) # whether there is true synchronization is hidden in pas_manager
            logging('iter '+str(iter_id)+' finish')

        epoch_id += 1 
        accuracy = test(test_loader, model)
        logging(' -- finish epoch: '+str(epoch_id) + ' -- | test accuracy: ' +str(accuracy) + '\n\n')

        if epoch_id % args.lr_decay_interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            apf_manager.tighter_stable_criterion()

if __name__ == "__main__":
    logging('Initialization:\n\t'
            + 'model: ' + args.model + '; dataset: ' + args.dataset + ';\n\t'
            + 'master_address: ' + str(args.master_address) + '; world_size: '+str(args.world_size) + '; rank: '+ str(args.rank) + ';\n\t'
            + 'initial_learning_rate: '+ str(args.initial_lr) +'; weight_decay: ' + str(args.weight_decay) + ';\n\t'
            + 'sync_frequency: '+ str(args.sync_frequency) +'; frozen_frequency: ' + str(args.frozen_frequency) + '; ema_alpha: ' + str(args.ema_alpha)+'; stable_threshold: ' + str(args.stable_threshold) + '\n')
    run()
