import random
import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F

class Stack(object):
    """栈"""
    def __init__(self, params):
        self.ry = -1 * torch.ones(1, dtype=torch.int64).cuda()
        self.re = -1 * torch.ones(1, dtype=torch.int64).cuda()
        # self.hidden_state = s
        self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        if self.items == []:
            return [self.ry, self.re]
        else:
            return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)


def gen_sample(model, x, params, worddicts_r, reworddicts, k=1):
    structures_1 = [16, 28, 64, 92, 146, 171, 181, 234, 250, 366]
    # structures_2 = [111, 267]
    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k
    # current living paths and corresponding scores(-log)
    hyp_samples = [] 
    hyp_scores = np.zeros(live_k).astype(np.float32)   
    
    next_state, ctx0 = model.f_init(x)    # ctx0: (1, 936, 8, 8)   next_state : (1, n)
    calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()   # (1, 8, 8)

    stack = Stack(params)
    s = next_state
    for i in range(params['maxlen']):

        item = stack.pop()
        ry = item[0]
        re = item[1]
        # s = item[2]

        # import pdb; pdb.set_trace()
        score, s, calpha_past = model.f_next(params, ry, re, ctx0, s, calpha_past)
        score = score[0,:,:]

        prob_y = F.softmax(score,1) #(batch,K)

        next_p = prob_y.detach().cpu().numpy()  # symbol probabilities, (live_k,K)
        next_state = s.detach().cpu().numpy()  # h2t, (live_k,n)
        cand_scores = hyp_scores[:, None] - np.log(next_p)  # (live_k, K)
        cand_flat = cand_scores.flatten()  # (live_k x K,)
        # live_k most likely paths

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]  # (k-dead_k,)
        voc_size = next_p.shape[1]  # K
        trans_indices = ranks_flat // voc_size  # yt-1, (k-dead_k,), maybe repeated
        word_indices = ranks_flat % voc_size  # yt, (k-dead_k,)
        costs = cand_flat[ranks_flat]  # path probabilities, (k-dead_k,)

        # update paths
        # import pdb; pdb.set_trace()
        wi = word_indices[0]
        hyp_samples.append(wi)
        hyp_scores = copy.deepcopy(costs)
        hyp_ry = wi * torch.ones(1, dtype=torch.int64).cuda()
        # hyp_state = copy.deepcopy(s)
        if wi in structures_1:
            # hyp_re1 = reworddicts[worddicts_r[hyp_ry] + '1']
            hyp_re1 = reworddicts['left']
            hyp_re1 = hyp_re1 * torch.ones(1, dtype=torch.int64).cuda()
            # hyp_re2 = reworddicts[worddicts_r[hyp_ry] + '2']
            hyp_re2 = reworddicts['right']
            hyp_re2 = hyp_re2 * torch.ones(1, dtype=torch.int64).cuda()
            stack.push([hyp_ry, hyp_re2])
            stack.push([hyp_ry, hyp_re1])
        if wi in [412]:
            # import pdb; pdb.set_trace()
            hyp_re = reworddicts['Start'] * torch.ones(1, dtype=torch.int64).cuda()
            stack.push([hyp_ry, hyp_re])

        if i > 0:
            if wi == 0 or stack.size() == 0:    # end
                break

    return hyp_scores, hyp_samples

# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    return lexicon