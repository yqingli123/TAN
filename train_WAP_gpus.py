from random import shuffle
import time
import os
import re
import numpy as np 
import random
import copy
import pickle as pkl
import torch
from torch import optim, nn
import torch.distributed as dist
from utils.distributed_triplet import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import load_dict, prepare_data, gen_sample, weight_init, compute_wer, compute_sacc
from model.encoder_decoder_v4 import Encoder_Decoder
from utils.data_iterator_lyq_v1_n4_modv1 import dataIterator, data_preprocessing
from utils.gtd_lyq import gtd2latex
from utils.utils import cmp_result

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True
# whether reload params
reload_flag = False

# load configurations
# root_paths
bfs_path = '/yrfs1/intern/yqli28/data/cuobiezi/right_chars/small_tree_decoder_v2_pkl_v1_newIDSv4_modv1/'
bfs_path_test = '/yrfs1/intern/yqli28/data/cuobiezi/right_chars/tree_decoder_v2_pkl_v1_newIDSv4_modv1/'
bfs2_path = '/yrfs1/intern/yqli28/data/cuobiezi/right_chars/small_tree_decoder_sv5_pkl_v1_newIDSv4_modv1/'
bfs2_path_test = '/yrfs1/intern/yqli28/data/cuobiezi/right_chars/tree_decoder_sv5_pkl_v1_newIDSv4_modv1/'
work_path = './'

model_idx = 1
# paths
dictionaries = [work_path + 'dictionary_object_newtree_sv5_newIDSv4.txt', 
                work_path + 'dictionary_relation_newtree_sv5_newIDSv4.txt', 
                work_path + 'dictionary_relation2_newtree_sv5_newIDSv4.txt']

datasets = [bfs2_path + 'train_imgs.pkl', 
            bfs2_path + 'train_labels.pkl', 
            bfs2_path + 'train_tlabels.pkl', 
            bfs_path + 'train_tlabels.pkl', 
            bfs2_path + 'train_aligns.pkl',
            bfs2_path + 'train_size.pkl']

valid_datasets = [bfs2_path_test + 'valid_imgs.pkl', 
                  bfs2_path_test + 'valid_labels.pkl', 
                  bfs2_path_test + 'valid_tlabels.pkl', 
                  bfs_path_test + 'valid_tlabels.pkl', 
                  bfs2_path_test + 'valid_aligns.pkl',
                  bfs2_path_test + 'valid_size.pkl']
                  
valid_output = [work_path+'result'+'/symbol_relation/', 
                work_path+'result'+'/memory_alpha/']
valid_result = [work_path+'result'+'/valid.cer', 
                work_path+'result'+'/valid.exprate']
saveto = work_path+'result'+'/WAP_params.pkl'
last_saveto = work_path+'result'+'/WAP_params_last.pkl'

# training settings
maxlen = 25
K = 6
batch_size = 96
max_epochs = 1000
lrate = 1.5
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 10
validStart = 0
finish_after = 10000000

# model architecture
params = {}
params['n'] = 256   # hidden_state dim
params['m'] = 256   # embedding dim
params['m_re'] = 256   # embedding dim
params['dim_attention'] = 512
params['D'] = 936
params['K'] = 413

params['Kre'] = 4
params['rre'] = 22
params['mre'] = 256
params['maxlen'] = maxlen

params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1

params['lpred_loss'] = 1.
params['rrepred_loss'] = 0.5
params['triplet'] = 0.05

# load dictionary
worddicts = load_dict(dictionaries[0])
if rank == 0:
    print ('total chars',len(worddicts), rank)
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

reworddicts = load_dict(dictionaries[1])
if rank == 0:
    print ('total relations',len(reworddicts), rank)
reworddicts_r = [None] * len(reworddicts)
for kk, vv in reworddicts.items():
    reworddicts_r[vv] = kk

rredicts = load_dict(dictionaries[2])
if rank == 0:
    print ('total chars',len(rredicts), rank)
rredicts_r = [None] * len(rredicts)
for kk, vv in rredicts.items():
    rredicts_r[vv] = kk

with open(valid_datasets[2], 'rb') as fp:
    groudtruth_gtd = pkl.load(fp)
with open(valid_datasets[0], 'rb') as fp:
    valid_imgs = pkl.load(fp)

valid_imgs_list = [(img, key) for key, img in valid_imgs.items()]   
def prepare_for_valid(batch):
    valid_img = batch[0][0]
    h, w = valid_img.shape
    valid_uid = batch[0][1]
    valid_img_tensor = torch.from_numpy(valid_img).to(torch.float32)

    valid_img_tensor = valid_img_tensor[None, None, :, :] / 255.   # normalization
    return valid_img_tensor, valid_uid

# 多卡分布式验证 --- 分布式采样
validSet = dataIterator(valid_datasets[0], valid_datasets[2], valid_datasets[3])
valid_preprocess = data_preprocessing(worddicts, reworddicts, rredicts)
with open(valid_datasets[1], 'rb') as f:
    valid_labels_dict = pkl.load(f)
valid_radical2chars_txt = './radical2chars_valid_processed_16.txt'
f = open(valid_radical2chars_txt, 'r')
lines = f.readlines()
f.close()
valid_pids_buckets = []
for line in lines:
    items = line.strip().split()
    radical = items[0]
    freq = items[1]
    unicodes_list = items[2:]
    valid_pids_buckets.append(unicodes_list)

valid_datasampler = MyDistributedSampler(validSet, valid_labels_dict, valid_pids_buckets, batch_size, 6, shuffle=False, num_replicas=world_size, rank=rank)
valid = DataLoader(validSet, batch_size=batch_size, sampler=valid_datasampler, collate_fn=valid_preprocess)

# 多卡分布式训练 --- 分布式采样
trainSet = dataIterator(datasets[0],  datasets[2], datasets[3])
preprocess = data_preprocessing(worddicts, reworddicts, rredicts)
with open(datasets[1], 'rb') as f:
    labels_dict = pkl.load(f)

radical2chars_txt = './radical2chars_processed_16.txt'
f = open(radical2chars_txt, 'r')
lines = f.readlines()
f.close()
pids_buckets = []
for line in lines:
    items = line.strip().split()
    radical = items[0]
    freq = items[1]
    unicodes_list = items[2:]
    pids_buckets.append(unicodes_list)

train_datasampler = MyDistributedSampler(trainSet, labels_dict, pids_buckets, batch_size, 6, shuffle=True, num_replicas=world_size, rank=rank)
train = DataLoader(trainSet, batch_size=batch_size, num_workers=2, sampler=train_datasampler, collate_fn=preprocess)

# display
dispFreq = 300


# initialize model
WAP_model = Encoder_Decoder(params)
if init_param_flag:
    WAP_model.apply(weight_init)
if reload_flag:
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    WAP_model.load_state_dict(torch.load(saveto, map_location=map_location))
# if multi_gpu_flag:
#     WAP_model = nn.DataParallel(WAP_model, device_ids=[0, 1, 2, 3])
WAP_model = WAP_model.cuda()
DDP_model = DDP(WAP_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
optimizer = optim.Adadelta(DDP_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)

if rank == 0:
    print('Optimization')

# statistics
history_loss_s = 100.
for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    uidx = 0  # count batch
    loss_s = 0.
    loss_lpred_s = 0.
    loss_rrepred_s = 0.
    loss_triplet_s = 0.
    valid_num_s = 0.
    valid_sum_s = 0.
    ud_s = 0  # time for training an epoch

    train_datasampler.set_epoch(eidx)

    if eidx == 0:
        sum_batch = len(train)
        if rank == 0:
            print('Numb train batch =', len(train))

    for x, x_mask, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask in train:
    
        DDP_model.train()
        ud_start = time.time()
        n_samples += len(x)
        uidx += 1
        # import pdb 
        # pdb.set_trace()
        x = x.float().cuda()
        x_mask = x_mask.float().cuda()
        ly = torch.from_numpy(ly).cuda()  # (seqs_y,batch)
        ly_mask = torch.from_numpy(ly_mask).cuda()  # (seqs_y,batch)
        ry = torch.from_numpy(ry).cuda()  # (seqs_y,batch)
        ry_mask = torch.from_numpy(ry_mask).cuda()  # (seqs_y,batch)
        lp = torch.from_numpy(lp).cuda()  # (seqs_y,batch)
        rp = torch.from_numpy(rp).cuda()  # (seqs_y,batch)
        re = torch.from_numpy(re).cuda()
        rre = torch.from_numpy(rre).cuda()  # (seqs_y,batch)
        rre_mask = torch.from_numpy(rre_mask).cuda()  # (seqs_y,batch)
        
        # forward
        loss, loss_lpred, loss_rrepred, loss_triplet, valid_num, valid_sum  = DDP_model(params, x, x_mask, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask)

        loss_s += loss.item()
        loss_lpred_s += loss_lpred.item()
        loss_rrepred_s += loss_rrepred.item()
        loss_triplet_s += loss_triplet.item()
        valid_num_s += valid_num.item()
        valid_sum_s += valid_sum.item()
        

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(DDP_model.parameters(), clip_c)
            
        # update
        optimizer.step()

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0 and rank == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            loss_lpred_s /= dispFreq
            loss_rrepred_s /= dispFreq
            loss_triplet_s /= dispFreq
            valid_percent = valid_num_s / valid_sum_s
            print('Epoch {}  Update {}  Cost {:.5f}  Cost lpred {:.5f}  Cost rre {:.5f} Cost triplet {:.5f}  valid percent {:.3f}  UD {:.3f}  lrate {}  eps {}  bad_counter {}' \
                .format(eidx, uidx, loss_s, loss_lpred_s, loss_rrepred_s, loss_triplet_s, valid_percent, ud_s, lrate, my_eps, bad_counter))
            ud_s = 0
            loss_s = 0.
            loss_lpred_s = 0.
            loss_rrepred_s = 0.
            loss_triplet_s = 0.
            valid_num_s = 0.
            valid_sum_s = 0.
            # break

    # validation
    if rank == 0:
        print('begin eval')
    ud_valid = time.time()

    total_number = 0
    n_batch = 0
    loss_valid_s = 0.
    loss_valid_lpred_s = 0.
    loss_valid_rrepred_s = 0.
    loss_valid_triplet_s = 0.
    validstage_valid_num_s = 0.
    validstage_valid_sum_s = 0.

    DDP_model.eval()
    with torch.no_grad():
        for x, x_mask, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask in valid:
            x = x.float().cuda()
            x_mask = x_mask.float().cuda()
            ly = torch.from_numpy(ly).cuda()  # (seqs_y,batch)
            ly_mask = torch.from_numpy(ly_mask).cuda()  # (seqs_y,batch)
            ry = torch.from_numpy(ry).cuda()  # (seqs_y,batch)
            ry_mask = torch.from_numpy(ry_mask).cuda()  # (seqs_y,batch)
            lp = torch.from_numpy(lp).cuda()  # (seqs_y,batch)
            rp = torch.from_numpy(rp).cuda()  # (seqs_y,batch)
            re = torch.from_numpy(re).cuda()  # (seqs_y,batch)
            rre = torch.from_numpy(rre).cuda()  # (seqs_y,batch)
            rre_mask = torch.from_numpy(rre_mask).cuda()  # (seqs_y,batch)

            loss, loss_lpred, loss_rrepred, loss_triplet, valid_num, valid_sum = DDP_model(params, x, x_mask, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask)

            loss_valid_s += loss.item()       
            loss_valid_lpred_s += loss_lpred.item()       
            loss_valid_rrepred_s += loss_rrepred.item()       
            loss_valid_triplet_s += loss_triplet.item()   
            validstage_valid_num_s += valid_num    
            validstage_valid_sum_s += valid_sum    
            n_batch += 1

    loss_valid_s /= n_batch 
    loss_valid_s = torch.tensor(loss_valid_s).cuda()
    dist.all_reduce(loss_valid_s, op=dist.ReduceOp.SUM)
    loss_valid_s = loss_valid_s / world_size

    loss_valid_lpred_s /= n_batch 
    loss_valid_lpred_s = torch.tensor(loss_valid_lpred_s).cuda()
    dist.all_reduce(loss_valid_lpred_s, op=dist.ReduceOp.SUM)
    loss_valid_lpred_s = loss_valid_lpred_s / world_size

    loss_valid_rrepred_s /= n_batch 
    loss_valid_rrepred_s = torch.tensor(loss_valid_rrepred_s).cuda()
    dist.all_reduce(loss_valid_rrepred_s, op=dist.ReduceOp.SUM)
    loss_valid_rrepred_s = loss_valid_rrepred_s / world_size

    loss_valid_triplet_s /= n_batch 
    loss_valid_triplet_s = torch.tensor(loss_valid_triplet_s).cuda()
    dist.all_reduce(loss_valid_triplet_s, op=dist.ReduceOp.SUM)
    loss_valid_triplet_s = loss_valid_triplet_s / world_size

    validstage_valid_num_s /= n_batch 
    validstage_valid_num_s = torch.tensor(validstage_valid_num_s).cuda()
    dist.all_reduce(validstage_valid_num_s, op=dist.ReduceOp.SUM)
    validstage_valid_num_s = validstage_valid_num_s / world_size

    validstage_valid_sum_s /= n_batch 
    validstage_valid_sum_s = torch.tensor(validstage_valid_sum_s).cuda()
    dist.all_reduce(validstage_valid_sum_s, op=dist.ReduceOp.SUM)
    validstage_valid_sum_s = validstage_valid_sum_s / world_size

    validstage_valid_percent = validstage_valid_num_s / validstage_valid_sum_s

    ud_valid = (time.time() - ud_valid) / 60.
    if rank == 0:
        print('valid set decode done, epoch cost time ... ', ud_valid)
        print('valid loss {:.5f}  valid loss {:.5f}  valid loss_rrepred {:.5f}  valid loss_triplet {:.5f}  valid_triplet_percent {:.3f} '.format(\
            loss_valid_s, loss_valid_lpred_s, loss_valid_rrepred_s, loss_valid_triplet_s, validstage_valid_percent))
           
    if rank == 0:
        print('Saving latest model params ... ')
        torch.save(DDP_model.module.state_dict(), last_saveto)

    # the first time validation or better model
    if  loss_valid_s <= history_loss_s :
        history_loss_s  = loss_valid_s
        bad_counter = 0
        if rank == 0:
            print('Saving best model params ... ')
            torch.save(DDP_model.module.state_dict(), saveto)

    # worse model
    if loss_valid_s > history_loss_s:
        bad_counter += 1
        if bad_counter > patience:
            if halfLrFlag == 3:   
                if rank == 0:          
                    print('Early Stop!')
                estop = True
                break
            else:
                print('Lr decay and retrain! in Rank:', rank)
                bad_counter = 0
                lrate = lrate / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lrate

                halfLrFlag += 1
                
    # finish after these many updates
    if uidx >= finish_after:
        print('Finishing after %d iterations!' % uidx)
        estop = True
        break
        
    # dist.barrier()
    if rank == 0:
        epoch_time = (time.time() - ud_epoch) / 60.
        print('epoch cost time  {:3f} min '.format(epoch_time))
            
    # early stop
    if estop:
        print('Rank %d Seen %d samples' % (rank, n_samples))
        break
