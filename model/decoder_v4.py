import torch
import torch.nn as nn
from utils.triplet_loss import *


# two layers of GRU
class Gru_cond_layer(nn.Module):
    def __init__(self, params):
        super(Gru_cond_layer, self).__init__()
        self.fc_Wyz0 = nn.Linear(params['m'], params['n'])
        self.fc_Wyr0 = nn.Linear(params['m'], params['n'])
        self.fc_Wyh0 = nn.Linear(params['m'], params['n'])
        self.fc_Uhz0 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr0 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh0 = nn.Linear(params['n'], params['n'], bias=False)

        # attention for parent symbol
        self.conv_UaP = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_WaP = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_QP = nn.Conv2d(1, 512, kernel_size=3, bias=False, padding=1)
        self.fc_UfP = nn.Linear(512, params['dim_attention'])
        self.fc_vaP = nn.Linear(params['dim_attention'], 1)

        # attention for memory
        # self.fc_Uamem = nn.Linear(params['m'], params['dim_attention'])
        # self.fc_Wamem = nn.Linear(params['D'], params['dim_attention'], bias=False)
        # # self.conv_Qmem = nn.Conv2d(1, 512, kernel_size=(3,1), bias=False, padding=(1,0))
        # # self.fc_Ufmem = nn.Linear(512, params['dim_attention'])
        # self.fc_vamem = nn.Linear(params['dim_attention'], 1)

        self.fc_Wyz1 = nn.Linear(params['D'], params['n'])
        self.fc_Wyr1 = nn.Linear(params['D'], params['n'])
        self.fc_Wyh1 = nn.Linear(params['D'], params['n'])
        self.fc_Uhz1 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr1 = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh1 = nn.Linear(params['n'], params['n'], bias=False)

        # the first GRU layer
        self.fc_Wyz = nn.Linear(2*params['m'], params['n'])
        self.fc_Wyr = nn.Linear(2*params['m'], params['n'])
        self.fc_Wyh = nn.Linear(2*params['m'], params['n'])

        self.fc_Uhz = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh = nn.Linear(params['n'], params['n'], bias=False)

        # attention for child symbol
        self.conv_UaC = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_WaC = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_QC = nn.Conv2d(1, 512, kernel_size=3, bias=False, padding=1)
        self.fc_UfC = nn.Linear(512, params['dim_attention'])
        self.fc_vaC = nn.Linear(params['dim_attention'], 1)
        self.fc_relation = nn.Linear(params['n'], params['dim_attention'], bias=False)

        # the second GRU layer
        self.fc_Wcz = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wcr = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wch = nn.Linear(params['D'], params['n'], bias=False)

        self.fc_Uhz2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhr2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhh2 = nn.Linear(params['n'], params['n'])

    def forward(self, params, rembedding, re_embedding, rp, ly_mask=None, 
        context=None, context_mask=None, init_state=None):
        
        n_steps = rembedding.shape[0]  # seqs_y
        n_samples = rembedding.shape[1]  # batch

        pctx_ = self.conv_UaC(context)  # (batch,n',H,W)
        pctx_ = pctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')

        emb = torch.cat((rembedding, re_embedding), 2)
        state_below_z = self.fc_Wyz(emb)
        state_below_r = self.fc_Wyr(emb)
        state_below_h = self.fc_Wyh(emb)
            
        calpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()  # (batch,H,W)
        # palpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()

        h2ts = torch.zeros(n_steps+1, n_samples, params['n']).cuda()
        h2ts[0, :, :] = init_state
        h2t = init_state

        ctCs = torch.zeros(n_steps+1, n_samples, params['D']).cuda()
        ctPs = torch.zeros(n_steps+1, n_samples, params['D']).cuda()
        calphas = torch.zeros(n_steps+1, n_samples, context.shape[2], context.shape[3]).cuda()
        calpha_pasts = torch.zeros(n_steps+1, n_samples, context.shape[2], context.shape[3]).cuda()

        for i in range(n_steps):
            rpos = rp[i]
            # h2t = torch.stack([h2ts[rpos[j], j, :] for j in range(n_samples)], 0)
            # import pdb; pdb.set_trace()
            ctP = torch.stack([ctCs[rpos[j], j, :] for j in range(n_samples)], 0)
            h2t, ctC, calpha, calpha_past = self._step_slice(ly_mask[i], 
                                                            context_mask, h2t, calpha_past, 
                                                            pctx_, context, state_below_z[i],
                                                            state_below_r[i], state_below_h[i])

            h2ts[i+1] = h2t  # (seqs_y,batch,n)
            ctCs[i+1] = ctC
            ctPs[i+1] = ctP
            calphas[i+1] = calpha  # (seqs_y,batch,H,W)
            calpha_pasts[i+1] = calpha_past  # (seqs_y,batch,H,W)

        return h2ts[1:], ctCs[1:], ctPs[1:], calphas[1:], calpha_pasts[1:]

    def valid_forward(self, params, rembedding, re_embedding, ry_mask=None, context=None, context_mask=None, init_state=None, alpha_past=None):
        emb = torch.cat((rembedding, re_embedding), 1)
        state_below_z = self.fc_Wyz(emb)
        state_below_r = self.fc_Wyr(emb)
        state_below_h = self.fc_Wyh(emb) 

        pctx_ = self.conv_UaC(context)  # (batch,n',H,W)
        pctx_ = pctx_.permute(2, 3, 0, 1)  # (H,W,batch,n')     

        if ry_mask is None:
            ry_mask = torch.ones(rembedding.shape[0]).cuda()

        h2t, ctC, alpha, alpha_past = self._step_slice(ry_mask, 
                                                        context_mask, init_state, alpha_past, 
                                                        pctx_, context, state_below_z,
                                                        state_below_r, state_below_h)  
        return h2t, ctC, alpha, alpha_past  


    # one step of two GRU layers
    def _step_slice(self, ly_mask, ctx_mask, h_, calpha_past_, 
        pctx_, cc_, state_below_lz, state_below_lr, state_below_lh):
        
        z0 = torch.sigmoid(self.fc_Uhz0(h_) + state_below_lz)  # (batch,n)
        r0 = torch.sigmoid(self.fc_Uhr0(h_) + state_below_lr)  # (batch,n)
        h0_p = torch.tanh(self.fc_Uhh0(h_) * r0 + state_below_lh)  # (batch,n)
        h0 = z0 * h_ + (1. - z0) * h0_p  # (batch,n)
        h0 = ly_mask[:, None] * h0 + (1. - ly_mask)[:, None] * h_   #  ~s_t^p

        # attention for child symbol
        # relation_information = self.fc_relation(re_embedding)   # relation information
        query_child = self.fc_WaC(h0)
        calpha_past__ = calpha_past_[:, None, :, :]  # (batch,1,H,W)
        cover_FC = self.conv_QC(calpha_past__).permute(2, 3, 0, 1)  # (H,W,batch,n')
        ccover_vector = self.fc_UfC(cover_FC)  # (H,W,batch,n')
        cattention_score = torch.tanh(pctx_ + query_child[None, None, :, :] + ccover_vector)  # (H,W,batch,n')
        # cattention_score = torch.tanh(pctx_ + query_child[None, None, :, :])  # (H,W,batch,n')
        calpha = self.fc_vaC(cattention_score)  # (H,W,batch,1)
        calpha = calpha - calpha.max()
        calpha = calpha.view(calpha.shape[0], calpha.shape[1], calpha.shape[2])  # (H,W,batch)
        calpha = torch.exp(calpha)  # exp
        if (ctx_mask is not None):
            calpha = calpha * ctx_mask.permute(1, 2, 0)
        calpha = (calpha / calpha.sum(1).sum(0)[None, None, :] + 1e-10)  # (H,W,batch)
        calpha_past = calpha_past_ + calpha.permute(2, 0, 1)  # (batch,H,W)
        ctC = (cc_ * calpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)  # current context, (batch,D)

        # the second GRU layer
        # ct = torch.cat((ctC, ctP), 1)
        z2 = torch.sigmoid(self.fc_Uhz2(h0) + self.fc_Wcz(ctC))  # zt  (batch,n)
        r2 = torch.sigmoid(self.fc_Uhr2(h0) + self.fc_Wcr(ctC))  # rt  (batch,n)
        h2_p = torch.tanh(self.fc_Uhh2(h0) * r2 + self.fc_Wch(ctC))  # (batch,n)
        h2 = z2 * h0 + (1. - z2) * h2_p  # (batch,n)
        h2 = ly_mask[:, None] * h2 + (1. - ly_mask)[:, None] * h0   # s_t^c

        return h2, ctC, calpha.permute(2, 0, 1), calpha_past



# calculate probabilities
class Gru_prob(nn.Module):
    def __init__(self, params):
        super(Gru_prob, self).__init__()
        self.fc_WctC = nn.Linear(params['D'], params['m'])
        self.fc_WhtC = nn.Linear(params['n'], params['m'])
        self.fc_WytC = nn.Linear(params['m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0C = nn.Linear(int(params['m'] / 2), params['K'])
        # self.fc_WctP = nn.Linear(params['D'], params['m'])
        self.fc_W0P = nn.Linear(int(params['m'] / 2), params['K'])
        self.fc_WctRe = nn.Linear(2*params['D'], params['mre'])
        self.fc_W0Re = nn.Linear(int(params['mre']), params['Kre'])

        self.triplet_loss_func = TripletLoss(margin=0.15)

    def forward(self, ctCs, htCs, prevC, y, use_dropout):
        clogit = self.fc_WctC(ctCs) + self.fc_WhtC(htCs) + self.fc_WytC(prevC)  # (seqs_y,batch,m)
        # clogit = self.fc_WctC(ctCs) + self.fc_WytC(prevC)  # (seqs_y,batch,m)
        cfeats = clogit.view(-1, clogit.shape[2])
        y_flatten = y.view(-1, 1)
        # import pdb; pdb.set_trace()
        triplet_loss, valid_num, valid_sum, semi_num, dist_ap, dist_an = self.triplet_loss_func(cfeats, y_flatten)
        if len(clogit.shape) == 2:
            clogit = clogit.unsqueeze(0)
        # maxout
        cshape = clogit.shape  # (seqs_y,batch,m)
        cshape2 = int(cshape[2] / 2)  # m/2
        cshape3 = 2
        clogit = clogit.view(cshape[0], cshape[1], cshape2, cshape3)  # (seqs_y,batch,m) -> (seqs_y,batch,m/2,2)
        clogit = clogit.max(3)[0]  # (seqs_y,batch,m/2)
        if use_dropout:
            clogit = self.dropout(clogit)
        cprob = self.fc_W0C(clogit)  # (seqs_y,batch,K)

        return cprob, triplet_loss, valid_num, valid_sum

    def forward_test(self, ctCs, htCs, prevC, use_dropout):
        clogit = self.fc_WctC(ctCs) + self.fc_WhtC(htCs) + self.fc_WytC(prevC)  # (seqs_y,batch,m)
        # cfeats = clogit
        cfeats = normalize(clogit)
        if len(clogit.shape) == 2:
            clogit = clogit.unsqueeze(0)
        # maxout
        cshape = clogit.shape  # (seqs_y,batch,m)
        cshape2 = int(cshape[2] / 2)  # m/2
        cshape3 = 2
        clogit = clogit.view(cshape[0], cshape[1], cshape2, cshape3)  # (seqs_y,batch,m) -> (seqs_y,batch,m/2,2)
        clogit = clogit.max(3)[0]  # (seqs_y,batch,m/2)
        if use_dropout:
            clogit = self.dropout(clogit)
        cprob = self.fc_W0C(clogit)  # (seqs_y,batch,K)

        return cprob, cfeats