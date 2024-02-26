# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

def orthognal_l2_reg(model:nn.Module, device):
    l2_reg = None
    for W in model.parameters():
        if W.ndimension() < 2:
            continue
        else:
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            if (rows > cols):
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols), requires_grad=True)
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(rows, rows), requires_grad=True)

            ident = ident.cuda(device)
            w_tmp = (m - ident)
            b_k = Variable(torch.rand(w_tmp.shape[1], 1))
            b_k = b_k.cuda(device)

            v1 = torch.matmul(w_tmp, b_k)
            norm1 = torch.norm(v1, 2)
            v2 = torch.div(v1, norm1)
            v3 = torch.matmul(w_tmp, v2)

            if l2_reg is None:
                l2_reg = (torch.norm(v3, 2)) ** 2
            else:
                l2_reg = l2_reg + (torch.norm(v3, 2)) ** 2
    return l2_reg