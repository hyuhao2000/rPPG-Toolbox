########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, config): # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters(): # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0 # extra scale for gain

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])


            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))


            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight) # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        with torch.no_grad(): # initial time_w curves for better convergence
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) # the distance
            for h in range(config.n_head):
                if h < config.n_head - 1:
                    decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        # if config.rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn(config)

        self.output = nn.Linear(config.n_attn, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # self.key.scale_init = 0
        # self.receptance.scale_init = 0
        # self.output.scale_init = 0

    def forward(self, x):
        # print(x[0][0][0])
        # print('time mixing')
        # print('time_alpha:',self.time_alpha.shape)
        # print('time_beta:',self.time_beta.shape)
        # print('time_gamma:',self.time_gamma.shape)

        B, T, C = x.size()
        # print('B:',B,'T:',T,'C:',C)
        TT = self.ctx_len
        # print('TT:',TT)
        w = F.pad(self.time_w, (0, TT))
        # print('w:',w.shape)
        w = torch.tile(w, [TT])
        # print('w:',w.shape)
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        # print('w:',w.shape)
        w = w[:, :, TT-1:] # w is now a circulant matrix
        # print('w:',w.shape)
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]
        # print('w:',w.shape)
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # print('x:',x.shape)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # print('k:',k.shape)
        # print('v:',v.shape)
        # print('r:',r.shape)
        k = torch.clamp(k, max=30, min=-60) # clamp extreme values. e^30 = 10^13
        # print('k:',k.shape) 
        k = torch.exp(k)
        # print('k:',k.shape)
        sum_k = torch.cumsum(k, dim=1)
        # print('sum_k:',sum_k.shape)
        kv = (k * v).view(B, T, self.n_head, self.head_size)
        # print('kv:',kv.shape)
        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)
        # print('wkv:',wkv.shape)
        rwkv = torch.sigmoid(r) * wkv / sum_k
        # print('rwkv:',rwkv.shape)
        rwkv = self.output(rwkv)
        # print('rwkv:',rwkv.shape)

        #mean
        mean = torch.mean(rwkv, dim=2)
        # print('mean:',mean.shape)
        # print('mean:',mean)
        grad = mean[:, 1:] - mean[:, :-1]
        # print('grad:',grad.shape)
        extend = torch.cat((grad, grad[:, -1:]), dim=1)
        # print('extend:',extend.shape)
        # print('extend:',extend)
        min_val, max_val = extend.min(), extend.max()
        # 归一化到 [0, 1] 之间
        normalized_grad = (extend - min_val) / (max_val - min_val)
        # 缩放到 [0.1, 0.2] 之间
        scaled_grad = normalized_grad + 0.5
        #取倒数
        # scaled_grad = (1 / scaled_grad)/10
        # print('scaled_grad:',scaled_grad.shape)
        # print('scaled_grad:',scaled_grad)
        # rwkv = rwkv * scaled_grad.unsqueeze(2)
        rwkv = self.dropout(rwkv)
        # print('rwkv:',rwkv.shape)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :],scaled_grad

class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * config.n_ffn // 2 # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # self.receptance.scale_init = 0
        # self.weight.scale_init = 0

    def forward(self, x, scaled_grad):
        # print('channel mixing')

        B, T, C = x.size()
        # print('B:',B,'T:',T,'C:',C)
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        # print('x:',x.shape)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # print('k:',k.shape)
        # print('v:',v.shape)
        # print('r:',r.shape)
        wkv = self.weight(F.mish(k) * v) # i find mish is a bit better than gelu
        # print('wkv:',wkv.shape)
        rwkv = torch.sigmoid(r) * wkv
        # print('rwkv:',rwkv.shape)
        rwkv = rwkv * scaled_grad.unsqueeze(2)
        rwkv = self.dropout(rwkv)
        # print('rwkv:',rwkv.shape)

        return rwkv



class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed

class FixedNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed

########################################################################################################

class RwkvConfig:
    def __init__(self, ctx_len,dropout=0.1, **kwargs):
        
        self.ctx_len = ctx_len
        self.dropout = dropout
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)


        self.attn = RWKV_TimeMix(config, layer_id)
        self.mlp = RWKV_ChannelMix(config, layer_id)


    def forward(self, x):
        
        tm, scaled_grad = self.attn(self.ln1(x))
        x = x + tm
        x = x + self.mlp(self.ln2(x),scaled_grad)
        
        return x

class Rwkv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv = nn.Conv3d(in_channels=6, out_channels=3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.encoder = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.decoder = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])

        self.ln_f1 = nn.LayerNorm(config.n_embd)
        self.ln_f2 = nn.LayerNorm(config.n_embd)

        self.time_out_1 = nn.Parameter(torch.ones(1,config.ctx_len,1)) # reduce confidence of early tokens
        self.time_out_2 = nn.Parameter(torch.ones(1,config.ctx_len,1))
        # self.head = nn.Linear(config.n_embd, 1, bias=False)

        self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        RWKV_Init(self, config)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (RMSNorm, nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('time' in fpn) or ('head' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, x):
        
        B, T, C, H, W = x.shape
        # print('x:',x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        # print('x:',x.shape)
        x = self.conv(x)
        # print('x:',x.shape)
        x = self.conv2(x)
        # print('x:',x.shape)
        x = x.squeeze(1)
        # print('x:',x.shape)
        
        x = x.view(B, T, -1)
    
        # print('x: ',x.shape)
        
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.encoder(x)
        x = self.ln_f1(x)
        x = x * self.time_out_1[:, :T, :]
        # x = self.head(x)
        #平均

        x = self.decoder(x)
        x = self.ln_f2(x)
        x = x * self.time_out_2[:, :T, :]

        x = torch.mean(x, dim=2).unsqueeze(2)

        return x
    
    
#main
if __name__ == '__main__':
    config = RwkvConfig(ctx_len=180,dropout=0,
                rwkv_emb_scale=0, 
                n_layer=1, n_head=4, 
                n_embd=72*72, 
                n_attn=4096, n_ffn=1024)
    model = Rwkv(config)
    # print(model)
    x = torch.randn(1, 180, 6, 72, 72)

    y = model(x)
    # print(y.shape)