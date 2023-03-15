from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class Triplet_Sampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, pids_buckets, batch_size, num_instances, shuffle=True):
        self.data_source = data_source
        self.pids_buckets = pids_buckets
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.shuffle = shuffle
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}

        # import pdb; pdb.set_trace()
        for index, pid in self.data_source.items():
            self.index_dic[pid].append(index)   

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):

        # shuffle K
        batch_idxs_dict = defaultdict(list)
        import pdb; pdb.set_trace()
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            if self.shuffle == True:
                random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avail_pids_buckets = copy.deepcopy(self.pids_buckets)
        final_idxs = []

        for i, bucket_list in enumerate(self.pids_buckets):
            buckets_set = list(set(bucket_list))
            if len(bucket_list) != len(buckets_set):
                print(i)
        # shuffle P
        while len(avail_pids_buckets[0]) >= self.num_pids_per_batch or len(avail_pids_buckets) > 1:
            if self.shuffle == True:
                random.shuffle(avail_pids_buckets)
            selected_pid_bucket = avail_pids_buckets[0]
            if len(selected_pid_bucket) >= self.num_pids_per_batch:
                selected_pids = random.sample(selected_pid_bucket, self.num_pids_per_batch)
                for pid in selected_pids:   # pid : 0x5412
                    # pid = pid[2:].upper()
                    # if self.shuffle == False:
                    # import pdb; pdb.set_trace()
                    try:
                        batch_idxs = batch_idxs_dict[pid[2:].upper()].pop(0)
                    except:
                        import pdb; pdb.set_trace()
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid[2:].upper()]) == 0:
                        for i, bucket in enumerate(avail_pids_buckets):
                            if pid in bucket:
                                # import pdb; pdb.set_trace()
                                bucket.remove(pid)
            else:

                avail_pids_buckets.remove(selected_pid_bucket)
                avail_pids_buckets[0].extend(selected_pid_bucket)
                avail_pids_buckets[0] = list(set(avail_pids_buckets[0]))

        return iter(final_idxs)


    def __len__(self):
        return self.length


class Triplet_List():
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, pids_buckets, batch_size, num_instances, shuffle=True):
        self.data_source = data_source
        self.pids_buckets = pids_buckets
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.shuffle = shuffle
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
    
        for index, pid in self.data_source.items():
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def forward(self):

        # shuffle K
        batch_idxs_dict = defaultdict(list)
        # import pdb; pdb.set_trace()
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            if self.shuffle == True:
                random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avail_pids_buckets = copy.deepcopy(self.pids_buckets)
        final_idxs = []

        for i, bucket_list in enumerate(self.pids_buckets):
            buckets_set = list(set(bucket_list))
            if len(bucket_list) != len(buckets_set):
                print(i)
        # shuffle P
        while len(avail_pids_buckets[0]) >= self.num_pids_per_batch or len(avail_pids_buckets) > 1:
            if self.shuffle == True:
                random.shuffle(avail_pids_buckets)
            selected_pid_bucket = avail_pids_buckets[0]
            if len(selected_pid_bucket) >= self.num_pids_per_batch:
                selected_pids = random.sample(selected_pid_bucket, self.num_pids_per_batch)
                for pid in selected_pids:

                    batch_idxs = batch_idxs_dict[pid[2:].upper()].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid[2:].upper()]) == 0:
                        # import pdb; pdb.set_trace()
                        for i, bucket in enumerate(avail_pids_buckets):
                            if pid in bucket:
                                # import pdb; pdb.set_trace()
                                bucket.remove(pid)
            else:

                avail_pids_buckets.remove(selected_pid_bucket)
                avail_pids_buckets[0].extend(selected_pid_bucket)
                avail_pids_buckets[0] = list(set(avail_pids_buckets[0]))

        return final_idxs

    def __len__(self):
        return self.length



class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

