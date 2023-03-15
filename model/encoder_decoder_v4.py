import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import DenseNet
from .decoder_v4 import Gru_cond_layer, Gru_prob
import math

# create gru init state
class FcLayer(nn.Module):
    def __init__(self, nin, nout):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        out = torch.tanh(self.fc(x))
        return out


# Embedding
class My_Embedding(nn.Module):
    def __init__(self, params):
        super(My_Embedding, self).__init__()
        self.embedding = nn.Embedding(params['K'], params['m'])
        self.re_embedding = nn.Embedding(params['Kre'], params['m'])
        self.pos_embedding = torch.zeros(params['maxlen'], params['m']).cuda()
        nin = params['maxlen']
        nout = params['m']
        d_model = nout
        for pos in range(nin):
            for i in range(nout//2):
                self.pos_embedding[pos, 2*i] = math.sin(1.*pos/(10000**(2.*i/d_model)))
                self.pos_embedding[pos, 2*i+1] = math.cos(1.*pos/(10000**(2.*i/d_model)))
    def forward(self, params, ly, lp, ry, re):
        if ly.sum() < 0.:  # <bos>
            lemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            lemb = self.embedding(ly)  # (seqs_y,batch,m)  |  (batch,m)
            if len(lemb.shape) == 3:  # only for training stage
                lemb_shifted = torch.zeros([lemb.shape[0], lemb.shape[1], params['m']], dtype=torch.float32).cuda()
                lemb_shifted[1:] = lemb[:-1]
                lemb = lemb_shifted

        if lp.sum() < 1.:  # pos=0
            Pemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            Pemb = self.pos_embedding[lp]  # (seqs_y,batch,m)  |  (batch,m)
            if len(Pemb.shape) == 3:  # only for training stage
                Pemb_shifted = torch.zeros([Pemb.shape[0], Pemb.shape[1], params['m']], dtype=torch.float32).cuda()
                Pemb_shifted[1:] = Pemb[:-1]
                Pemb = Pemb_shifted

        if ry.sum() < 0.:  # <bos>
            remb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            remb = self.embedding(ry)  # (seqs_y,batch,m)  |  (batch,m)
            if len(remb.shape) == 3:  # only for training stage
                remb_shifted = torch.zeros([remb.shape[0], remb.shape[1], params['m']], dtype=torch.float32).cuda()
                remb_shifted[1:] = remb[1:]
                remb = remb_shifted
        # import pdb; pdb.set_trace()
        if re.sum() < 0:
            re_emb = torch.zeros(1, params['m']).cuda()
        else:
            re_emb = self.re_embedding(re)
            if len(re_emb.shape) == 3:
                re_emb_shifted = torch.zeros([re_emb.shape[0], re_emb.shape[1], params['m']], dtype=torch.float32).cuda()
                re_emb_shifted[1:] = re_emb[1:]
                re_emb = re_emb_shifted

        return lemb, Pemb, remb, re_emb
        
    def re_emb(self, params, re):
        if re.sum() < 0:
            re_emb = torch.zeros(1, params['m']).cuda()
        else:
            re_emb = self.re_embedding(re)
        return re_emb
    def word_emb(self, params, y):
        if y.sum() < 0.:  # <bos>
            emb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            emb = self.embedding(y)  # (seqs_y,batch,m)  |  (batch,m)
        return emb
    def pos_emb(self, params, p):
        if p.sum() < 1.:  # <bos>
            Pemb = torch.zeros(1, params['m']).cuda()  # (1,m)
        else:
            Pemb = self.pos_embedding[p]  # (seqs_y,batch,m)  |  (batch,m)
        return Pemb

class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], reduction=params['reduction'],
                                bottleneck=params['bottleneck'], use_dropout=params['use_dropout'])
        self.init_GRU_model = FcLayer(params['D'], params['n'])
        self.emb_model = My_Embedding(params)
        self.gru_model = Gru_cond_layer(params)
        self.gru_prob_model = Gru_prob(params)
        self.fc_Uamem = nn.Linear(params['n'], params['dim_attention'])
        self.fc_Wamem = nn.Linear(params['n'], params['dim_attention'], bias=False)
        # self.conv_Qmem = nn.Conv2d(1, 512, kernel_size=(3,1), bias=False, padding=(1,0))
        # self.fc_Ufmem = nn.Linear(512, params['dim_attention'])
        self.fc_vamem = nn.Linear(params['dim_attention'], 1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.Wre = nn.Linear(2*params['D'], params['rre'])

    def forward(self, params, x, x_mask, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask, one_step=False):

        ctx, ctx_mask = self.encoder(x, x_mask) # ctx.shape=[batch, c, h, w]   ctx_mask.shape=[batch, h, w]

        # init state
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]  # (batch,D)
        init_state = self.init_GRU_model(ctx_mean)  # (batch,n)

        # two GRU layers
        # import pdb; pdb.set_trace()
        lemb, Pemb, remb, re_emb = self.emb_model(params, ly, lp, ry, re)  # (seqs_y,batch,m)
        # h2ts: (seqs_y,batch,n),  cts: (seqs_y,batch,D),  alphas: (seqs_y,batch,H,W)
        h2ts, ctCs, ctPs, calphas, calpha_pasts = \
                                self.gru_model(params, remb, re_emb, rp, ly_mask, ctx, ctx_mask, init_state=init_state)
        # ctCs [seq, batch, dim]
        # import pdb; pdb.set_trace()
        pred_re_scores = self.Wre(torch.cat((ctPs, ctCs), 2))
        cscores, triplet_loss, valid_num, valid_sum = self.gru_prob_model(ctCs, h2ts, remb, ly, use_dropout=params['use_dropout'])  # (seqs_y, batch, K)

        pred_re_scores = pred_re_scores.contiguous()
        cscores = cscores.contiguous()
        cscores_flatten = cscores.view(-1, cscores.shape[2])    # (seqs_y x batch,K)
        pred_re_scores_flatten = pred_re_scores.view(-1, pred_re_scores.shape[2])    # (seqs_y x batch,K)
        ly = ly.contiguous()
        rre = rre.contiguous()
        lpred_loss = self.criterion(cscores_flatten, ly.view(-1))  # (seqs_y x batch,)
        lpred_loss = lpred_loss.view(ly.shape[0], ly.shape[1])  # (seqs_y,batch)
        lpred_loss = (lpred_loss * ly_mask).sum(0) / (ly_mask.sum(0)+1e-10)
        lpred_loss = lpred_loss.mean()

        rre_pred_loss = self.criterion(pred_re_scores_flatten, rre.view(-1))  # (seqs_y x batch,)
        rre_pred_loss = rre_pred_loss.view(rre.shape[0], rre.shape[1])  # (seqs_y,batch)
        rre_pred_loss = (rre_pred_loss * rre_mask).sum(0) / (rre_mask.sum(0)+1e-10)
        rre_pred_loss = rre_pred_loss.mean()    

        loss = params['lpred_loss'] * lpred_loss + params['rrepred_loss'] * rre_pred_loss + params['triplet'] * triplet_loss

        # loss = params['ly_lambda'] * lpred_loss + params['triplet'] * triplet_loss
        
        return loss, lpred_loss, rre_pred_loss, triplet_loss, valid_num, valid_sum


    # decoding: encoder part
    def f_init(self, x, x_mask=None):
        if x_mask is None:  # x_mask is actually no use here
            shape = x.shape
            x_mask = torch.ones(shape).cuda()
        ctx, _ctx_mask = self.encoder(x, x_mask)
        ctx_mean = ctx.mean(dim=3).mean(dim=2)
        init_state = self.init_GRU_model(ctx_mean)  # (1,n)
        return init_state, ctx

    def f_next(self, params, ry, re, ctx0, hidden_state, alpha_past):
        remb = self.emb_model.word_emb(params, ry)
        re_emb = self.emb_model.re_emb(params, re)
        h2t, ctC, alpha, alpha_past = self.gru_model.valid_forward(params, remb, re_emb, context=ctx0, init_state=hidden_state, alpha_past=alpha_past)
        cscore, cfeat = self.gru_prob_model.forward_test(ctC, h2t, remb, use_dropout=params['use_dropout'])

        return cscore, h2t, alpha_past

    def f_next_attention(self, params, ry, re, ctx0, hidden_state, alpha_past):
        remb = self.emb_model.word_emb(params, ry)
        re_emb = self.emb_model.re_emb(params, re)
        h2t, ctC, alpha, alpha_past = self.gru_model.valid_forward(params, remb, re_emb, context=ctx0, init_state=hidden_state, alpha_past=alpha_past)
        cscore, cfeat = self.gru_prob_model.forward_test(ctC, h2t, remb, use_dropout=params['use_dropout'])

        return cscore, h2t, alpha, alpha_past

    def f_next_parent(self, params, ly, lp, ctx, init_state, h1t, palpha_past, nextemb_memory, nextePmb_memory, initIdx):
        emb = self.emb_model.word_emb(params, ly)
        # Pemb = self.emb_model.pos_emb(params, lp)
        nextemb_memory[initIdx, :, :] = emb
        # ePmb_memory_ = emb + Pemb
        nextePmb_memory[initIdx, :, :] = init_state

        h01, ctP, palpha, next_palpha_past = self.gru_model.parent_forward(params, emb, context=ctx, init_state=init_state, palpha_past=palpha_past)

        mempctx_ = self.fc_Uamem(nextePmb_memory)
        memquery = self.fc_Wamem(h01)
        memattention_score = torch.tanh(mempctx_ + memquery[None, :, :])
        memalpha = self.fc_vamem(memattention_score)
        memalpha = memalpha - memalpha.max()
        memalpha = memalpha.view(memalpha.shape[0], memalpha.shape[1]) # Matt * batch
        memalpha = torch.exp(memalpha)
        mem_mask = torch.zeros(nextePmb_memory.shape[0], nextePmb_memory.shape[1]).cuda()
        mem_mask[:(initIdx+1), :] = 1
        memalpha = memalpha * mem_mask # Matt * batch
        memalpha = memalpha / (memalpha.sum(0) + 1e-10)

        Pmemalpha = memalpha.view(-1, memalpha.shape[1])
        Pmemalpha = Pmemalpha.permute(1, 0) # batch * Matt
        return h01, Pmemalpha, ctP, palpha, next_palpha_past, nextemb_memory, nextePmb_memory

    # decoding: decoder part
    def f_next_child(self, params, remb, ctP, ctx, init_state, calpha_past):

        next_state, h1t, ctC, ctP, ct, calpha, next_calpha_past = \
                self.gru_model.child_forward(params, remb, ctP, context=ctx, init_state=init_state, calpha_past=calpha_past)

        # reshape to suit GRU step code
        h2te = next_state.view(1, next_state.shape[0], next_state.shape[1])
        ctC = ctC.view(1, ctC.shape[0], ctC.shape[1])
        ctP = ctP.view(1, ctP.shape[0], ctP.shape[1])
        ct = ct.view(1, ct.shape[0], ct.shape[1])

        # calculate probabilities
        cscores, pscores, rescores = self.gru_prob_model(ctC, ctP, ct, h2te, remb, use_dropout=params['use_dropout'])
        cscores = cscores.view(-1, cscores.shape[2])
        next_lprobs = F.softmax(cscores, dim=1)
        rescores = rescores.view(-1, rescores.shape[2])
        next_reprobs = F.softmax(rescores, dim=1)
        next_re = torch.argmax(next_reprobs, dim=1)

        return next_lprobs, next_reprobs, next_state, h1t, calpha, next_calpha_past, next_re
