import torch
import sys, os, argparse, copy, numpy, pprint, datetime, time

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

CUDA = torch.cuda.is_available()

class EdgeAggregationService(Service):
    def __init__(self, world_size, server_ip, server_port, edge_rank):
        self.world_size = world_size
        self.edge_rank = edge_rank
        self.edge_round_id = 0
        self.aggregated_tensor_list = None

        self.tensor_lists_collected = [0] * world_size
        self.has_arrived = [0] * world_size

        ''' connect to the FL server '''
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        self.connection = rpyc.connect(server_ip, server_port, config=rpyc_config)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Edge] '+str(string))
        sys.stdout.flush()

    def exposed_aggregate(self, rank, round_id, tensor_list):
        self.logging('receive rank %d, round_id %d' % (rank, round_id))
        self.tensor_lists_collected[rank] = rpyc.classic.obtain(tensor_list)
        self.has_arrived[rank] = 1
        self.edge_round_id = max(self.edge_round_id, round_id)

        ''' handle the case when all the clients have reported '''
        if sum(self.has_arrived) == self.world_size:
            ''' make aggregation and conduct potential analysis '''
            edge_aggregated_tensor_list = self.calculate_model_average()

            ''' report the local average to the global server '''
            self.aggregated_tensor_list = self.sync_with_global_server(edge_aggregated_tensor_list)

            ''' reset '''
            self.has_arrived = [0] * self.world_size
            self.tensor_lists_collected = [0] * self.world_size

        ''' otherwise, keep spining (blocked) '''
        while self.has_arrived[rank] == 1:
            time.sleep(0.01)

        self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.aggregated_tensor_list

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

    def sync_with_global_server(self, edge_aggregated_tensor_list):
        tensor_list_received = self.connection.root.aggregate(self.edge_rank, self.edge_round_id, edge_aggregated_tensor_list)
        global_tensor_list = [rpyc.classic.obtain(i).cuda() for i in tensor_list_received] if CUDA \
            else [rpyc.classic.obtain(i) for i in tensor_list_received]
        return global_tensor_list


if __name__ == "__main__":

    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=20000)
    parser.add_argument('--edge_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--client_number', type=int, default=5)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')

    args = parser.parse_args()
    print('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks + '\n')

    service = EdgeAggregationService(args.client_number, args.server_ip, args.server_port, args.edge_rank)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    edge = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    edge.start()