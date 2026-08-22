import torch
from torch.utils.data import BatchSampler, Sampler, DistributedSampler, DataLoader
import random
from collections import defaultdict
import math



class SameSessionBatchSamplerDDP(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False, rank=0, world_size=1, session_ids=None, session_lengths=None):
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0  
        if session_ids is None:
            session_ids = {}
            for idx, item in enumerate(dataset):
                
                session_ids[idx] = item[2] 
            
            self.session_ids = session_ids
        else:
            self.session_ids = session_ids

        
        self.num_samples = len(dataset)

    def __iter__(self):
        indices_for_this_rank = list(self.sampler) 
        session_batches_indices = defaultdict(list)
        for idx in indices_for_this_rank:
            session_id = self.session_ids[idx]
            session_batches_indices[session_id].append(idx)
            
        all_batches_for_rank = []
        for session_id, indices in session_batches_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches_for_rank.append(batch)
                
        random.seed(self.epoch * self.world_size + self.rank)
        random.shuffle(all_batches_for_rank)
        
        for batch in all_batches_for_rank:
            yield batch

    def __len__(self):
       
        num_samples_per_rank = self.num_samples // self.world_size
        if self.num_samples % self.world_size > self.rank:
            num_samples_per_rank += 1
            
        if self.drop_last:
            return num_samples_per_rank // self.batch_size
        else:
            return math.ceil(num_samples_per_rank / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
