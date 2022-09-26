import torch
import sys, os, argparse, copy, numpy, pprint, datetime

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager
from sync_manager import Sync_Manager

CUDA = torch.cuda.is_available()

class Client:
    def __init__(self, local_profile, sync_profile):
        """ Variables required for local training. """
        self.rank = local_profile['rank']
        
        ''' Model Initialization '''
        self.model_manager = Model_Manager(local_profile['model_profile'])
        self.model = self.model_manager.load_model()
        self.optimizer = self.model_manager.get_optimizer()
        self.loss_func = self.model_manager.get_loss_func()

        ''' Dataset Initialization ''' 
        self.dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
        self.training_dataloader = self.dataset_manager.get_training_dataloader()
        self.testing_dataloader = self.dataset_manager.get_testing_dataloader()
        self.max_epoch = 10000

        ''' Sync-related variables (the above part shall be able to run under local mode)'''
        self.sync_manager = Sync_Manager(self.model, self.rank, sync_profile)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client] '+str(string))
        sys.stdout.flush()
    
    def test(self):
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(self.testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_output = self.model(test_x)
                pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y) 
        accuracy = positive_test_number / total_test_number
        return accuracy

    def train(self):
        print('\n\n\t-------- START TRAINING --------\n')
        # self.logging(list(self.model.parameters())[0][0][0].data)
        iter_id, round_id, epoch_id  = 0, 0, 0
        while epoch_id < self.max_epoch:
            epoch_id += 1 
            self.logging('start epoch: %d' % epoch_id)
            for step, (b_x, b_y) in enumerate(self.training_dataloader):
                # self.logging('start iteration: %d' % step)
                iter_id += 1
                if CUDA:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                self.optimizer.zero_grad()
                self.loss_func(self.model(b_x), b_y).backward()
                self.optimizer.step()
                # self.logging('finish local iteration: %d' % step)
                if self.sync_manager.try_sync_model(iter_id):
                    # self.logging('finish try_sync: %d' % step)
                    round_id += 1
                    if self.rank == 0:
                        accuracy = self.test()
                        self.logging(' - test - iter_id: %d; epoch_id: %d, round_id: %d; accuracy: %.4f;' % (iter_id, epoch_id, self.sync_manager.sync_round_id, accuracy))
                        # numpy.save('/root/adaptive_freezing/vgg-npy/param_round_%d' % round_id, list(self.model.parameters())[0][0].detach().cpu().numpy())
                # self.logging('finish iteration: %d' % step)
            self.logging('finish epoch: %d\n' % epoch_id)


if __name__ == "__main__":

    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=5)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    if CUDA:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())

    ''' A. Local Training Profile '''
    model_name, dataset_name, is_iid = 'VGG16_Cifar10', 'Cifar10', True
    model_name, dataset_name, is_iid = 'LSTM_KWS', 'KWS', True
    model_name, dataset_name, is_iid = 'ResNet18_Cifar10', 'Cifar10', True
    model_name, dataset_name, is_iid = 'CNN_Cifar10', 'Cifar10', True
    local_profile = {
        'rank' : args.rank,
        'model_profile' : {
            'model_name' : model_name,
        },
        'dataset_profile' : {
            'dataset_name' : dataset_name,
            'is_iid' : is_iid,
            'total_partition_number' : 1 if is_iid else args.world_size,
            'partition_rank' : 0 if is_iid else args.rank
        }
    }
    print('- Local Training Profile: ')
    pprint.pprint(local_profile)
    print

    ''' B. Synchronization Profile '''
    sync_frequency = 100
    interlayer_type = 'CMFL'  # Default, APF, Gaia, CMFL
    interlayer_type = 'Gaia'  # Default, APF, Gaia, CMFL
    interlayer_type = 'APF'  # Default, APF, Gaia, CMFL
    interlayer_type = 'Default'  # Default, APF, Gaia, CMFL
    sync_profile = {
        'sync_frequency' : sync_frequency,
        'interlayer_type' : interlayer_type,
        'server_ip': args.server_ip,
        'server_port': args.server_port,
    }
    print('- Sync Profile: ')
    pprint.pprint(sync_profile)
    print 

    ''' Launch Training '''
    client = Client(local_profile, sync_profile) # prepare local training environment
    client.train() # prepare local training environment to specify synchronization 