import torch
import copy
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()

class APF_Manager:
    def __init__(self, model, master_address, world_size, rank, sync_frequency=1, frozen_frequency=250, ema_alpha=0.95, stable_threshold=0.1):
        if CUDA:
            dist.init_process_group(backend='nccl', init_method=master_address, world_size=world_size, rank=rank)
            torch.cuda.set_device(rank)
        else:
            dist.init_process_group(backend='tcp', init_method=master_address, world_size=world_size, rank=rank)
        group = dist.new_group([i for i in range(world_size)])

        for param in model.parameters():
            dist.broadcast(param.data, src=0, group=group)

        self.sync_frequency = sync_frequency
        self.frozen_frequency = frozen_frequency
        self.ema_alpha = ema_alpha 
        self.global_ac_grad_ema_threshold = stable_threshold # this shall be larger than 1-self.ema_alpha
        self.tighter_stable_criterion_threshold = 0.8

        self.frag_shape_list = []
        self.frag_index_list = []
        self.group = group
        self.world_size = world_size
        self.model = model
        self.round_id = 0

        self.null_tensor = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        s_index = 0
        flattened_param = self.null_tensor
        for p in model.parameters():
            self.frag_shape_list.append(p.data.shape)
            frag = p.data.view(-1)
            e_index = s_index + len(frag)
            self.frag_index_list.append([s_index, e_index])
            s_index = e_index
            flattened_param = torch.cat((flattened_param, frag),0)
        self.last_flattened_param = copy.deepcopy(flattened_param)
        self.model_size = len(flattened_param)
        self.flattened_shape = flattened_param.shape
        self.synchronization_mask = torch.ones(self.flattened_shape).byte().cuda() if CUDA else torch.ones(self.flattened_shape).byte()

        self.global_ac_grad_ema = torch.zeros(self.flattened_shape).cuda() if CUDA else torch.zeros(self.flattened_shape)
        self.global_abs_ac_grad_ema = torch.zeros(self.flattened_shape).cuda() if CUDA else torch.zeros(self.flattened_shape)
        self.frozen_lengths = torch.zeros(self.flattened_shape).int().cuda() if CUDA else torch.zeros(self.flattened_shape).int()
        self.defrozen_round_ids = torch.zeros(self.flattened_shape).int().cuda() if CUDA else torch.zeros(self.flattened_shape).int()

    def tighter_stable_criterion(self):
        self.global_ac_grad_ema_threshold /= 2.0
        print('make stable criterion tighter')

    def update_frozen_lengths(self, current_flattened_param):
        # update the synchronization mask of each parameter based on global gradient stability
        self.global_ac_grad = self.last_flattened_param - current_flattened_param
        active_index = self.synchronization_mask.nonzero()
        # print 'len of active index', len(active_index)
        self.global_ac_grad_ema[active_index] = self.global_ac_grad_ema[active_index] * self.ema_alpha + self.global_ac_grad[active_index] * (1.0 - self.ema_alpha) 
        self.global_abs_ac_grad_ema[active_index] = self.global_abs_ac_grad_ema[active_index] * self.ema_alpha + torch.abs(self.global_ac_grad[active_index]) * (1.0 - self.ema_alpha)
        self.frozen_lengths[active_index] = torch.where(torch.abs(self.global_ac_grad_ema[active_index]) / self.global_abs_ac_grad_ema[active_index] < self.global_ac_grad_ema_threshold, self.frozen_lengths[active_index]+1, self.frozen_lengths[active_index]/2)
        self.defrozen_round_ids[active_index] = self.round_id + self.frozen_lengths[active_index] + 1

    def sync(self, iter_id):
        flattened_param = self.null_tensor
        for p in self.model.parameters():
            frag = p.data.view(-1)
            flattened_param = torch.cat((flattened_param, frag),0)
        flattened_param = torch.where(self.synchronization_mask > 0, flattened_param, self.last_flattened_param)

        if iter_id % self.sync_frequency == 0:
            transmitted_param = torch.masked_select(flattened_param, self.synchronization_mask)
            print('prepare transmitting in iter: '+str(iter_id))
            dist.all_reduce(transmitted_param, op=dist.reduce_op.SUM, group=self.group)
            transmitted_param /= self.world_size
            flattened_param[self.synchronization_mask] = transmitted_param

            if iter_id % self.frozen_frequency == 0:
                # current for simplicity the full model is transmitted, but it can be compressed later
                self.update_frozen_lengths(flattened_param)
                self.last_flattened_param = copy.deepcopy(flattened_param)
    
                # update round id and defrozen corresponded parameters
                self.round_id += 1
                self.synchronization_mask = self.defrozen_round_ids == self.round_id
                stable_ratio = 1 - self.synchronization_mask.sum().float() / self.model_size
                # adjust the synchronization frequency when necessary
                # print 'at iteration: ', iter_id, '; round id: ', self.round_id, '; stable ratio: ', stable_ratio
                if stable_ratio > self.tighter_stable_criterion_threshold:
                    self.tighter_stable_criterion()

        for i, p in enumerate(self.model.parameters()):
            p.data = flattened_param[self.frag_index_list[i][0]:self.frag_index_list[i][1]].view(self.frag_shape_list[i])

