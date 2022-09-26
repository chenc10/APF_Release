import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

CUDA = torch.cuda.is_available()

class AggregationService(Service):
    def __init__(self, world_size):
        self.world_size = world_size
        self.tensor_lists_collected = [0] * world_size
        self.has_arrived = [0] * world_size

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def exposed_aggregate(self, rank, round_id, tensor_list):
        self.logging('receive rank %d, round_id %d' % (rank, round_id))
        self.tensor_lists_collected[rank] = rpyc.classic.obtain(tensor_list)
        self.has_arrived[rank] = 1

        ''' handle the case when all the clients have reported '''
        if sum(self.has_arrived) == self.world_size:
            ''' make aggregation and conduct potential analysis '''
            aggregated_tensor_list = self.calculate_model_average()

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_lists_collected = [0] * self.world_size

        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        self.logging('return rank %d, round_id %d' % (rank, round_id))
        return aggregated_tensor_list

    def calculate_model_average(self):
        aggregated_tensor_list = []
        for tensor_id, tensor_content in enumerate(self.tensor_lists_collected[0]):
            # calculate average layer by layer
            sum_tensor = torch.zeros(tensor_content.size())
            for rank in range(self.world_size):
                sum_tensor += self.tensor_lists_collected[rank][tensor_id]
            aggregated_tensor = sum_tensor / self.world_size
            aggregated_tensor_list.append(aggregated_tensor)
        
        return aggregated_tensor_list

if __name__ == "__main__":

    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=5)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    service = AggregationService(args.world_size)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    server = ThreadedServer(service, port=args.server_port, protocol_config=rpyc_config)
    server.start()