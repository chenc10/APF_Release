import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

CUDA = torch.cuda.is_available()

class AggregationService(Service):
    def __init__(self, edge_number):
        self.relaxation_factor = 0.9
        self.edge_number = edge_number
        self.global_tensor_list = None

        self.last_pushed_tensor_lists = [None] * edge_number # can be deleted later
        self.last_pull_times = [0] * edge_number

        self.latest_round_times = [0] * edge_number
        self.ema_max_round_time = 0 # current weight: 0.1
        self.pull_gap_time = 0

        self.next_pull_time = 0
        self.next_pull_rank = 0

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_aggregate(self, rank, round_id, tensor_list):

        ''' record push content and push time '''
        tensor_list = rpyc.classic.obtain(tensor_list)
        push_time = time.time()
        if round_id > 3: # avoid boundary case (the round_time would be inaccurate in the early stage)
            self.latest_round_times[rank] = push_time - self.last_pull_times[rank]
        self.logging('receive tensor from rank %d, round_id %d, latest_round_time: %.2f' % (rank, round_id, self.latest_round_times[rank]))

        ''' keep spining if the arriving edge does not get its turn '''
        while time.time() < self.next_pull_time or rank != self.next_pull_rank:
            time.sleep(0.01)
        
        ''' At its turn, update the model parameters '''
        if self.last_pull_times[rank] == 0: # boundary case: init the local model on all the clients
            if rank == 0: # set the first client's local model as the global one and return
                self.global_tensor_list = tensor_list
            self.last_pushed_tensor_lists[rank] = copy.deepcopy(self.global_tensor_list)
        else: # regular case: update the global model with the reported local gradient
            gradient_list = [tensor_list[i]-self.last_pushed_tensor_lists[rank][i] for i in range(len(tensor_list))]
            for tensor_id, tensor_content in enumerate(self.global_tensor_list):
                self.global_tensor_list[tensor_id] += 1.0/self.edge_number * gradient_list[tensor_id]
            self.last_pushed_tensor_lists[rank] = tensor_list
        
        ''' update pull_gap_time and next_pull_time, then hand over pull-turn to the next edge '''
        pull_time = time.time()
        self.last_pull_times[rank] = pull_time
        if rank == self.edge_number-1: # update max_round_time after the last edge pushes
            max_round_time = max(self.latest_round_times)
            self.ema_max_round_time = max_round_time if self.ema_max_round_time == 0 \
                else self.ema_max_round_time * 0.9 + max_round_time * 0.1
            self.pull_gap_time = self.ema_max_round_time / float(self.edge_number) * self.relaxation_factor
            self.logging('latest_round_times: %s, pull_gap_time updated to %.2f' % (self.latest_round_times, self.pull_gap_time))

        self.next_pull_time = pull_time + self.pull_gap_time
        self.next_pull_rank = (rank + 1) % self.edge_number

        self.logging('return rank %d, round_id %d\n' % (rank, round_id))
        return self.global_tensor_list

if __name__ == "__main__":

    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--edge_number', type=int, default=5)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    service = AggregationService(args.edge_number)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()