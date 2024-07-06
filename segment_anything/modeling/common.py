# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

class ConvBlock(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU):
        super().__init__()
        self.kernel_size = kernel_size
        #self.fc1 = nn.Linear(D_features, r_features)
        #self.fc2 = nn.Linear(r_features,D_features)
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
        #self.act = act_layer()
        self.conv_l_1 = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
        self.conv_l_2 = nn.Conv2d(r_features,
                      r_features,
                      kernel_size = kernel_size,
                      padding = 'same',
                      bias = True)
        
        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)


    def forward(self, x):
        # x : (B, window_size, window_size, embed_dim)
        #x = self.fc1(x)
        x = self.conv1(x.permute(0,3,1,2))
        #x = self.act(x)
        x = nn.functional.interpolate(x,scale_factor = 2,mode='bilinear')
        x = self.conv_l_1(x)
        x = self.conv_l_2(x)
        x = nn.functional.interpolate(x,scale_factor = 0.5, mode = 'bilinear')
        #x = self.conv(x.permute(0,3,1,2))
        #x = self.fc2(x.permute(0,2,3,1))
        x = self.conv2(x)
        return x.permute(0,2,3,1)

class Conv_Scale_Block(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU, num_experts = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
        self.conv_l_1 = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
        

        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)
        self.avg_pool = nn.AvgPool2d(kernel_size=64)
        self.H = nn.Linear(16,self.num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x : (B, window_size, window_size, embed_dim)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        pooled_x = self.avg_pool(x) # average pooling x, shape (B,r,1,1)
        reshaped_pooled_x = torch.reshape(pooled_x,(pooled_x.shape[:2])) # reshaped pooled_x, shape (B,r)
        H_out = self.H(reshaped_pooled_x) # H_out = H(x) shape (B,num_experts)
        _,indices = torch.sort(H_out,dim=1,descending=True)
        H_filtered = torch.clone(H_out)
        
        for b in range(indices.shape[0]):
            for ind in indices[b][2:]:
                H_filtered[b][ind] = -torch.inf
                
        G_out = self.softmax(H_filtered)  # G_out = G(x) shaped (B,num_experts)
        E = []
        
        for i in range(self.num_experts):
            
            xs = nn.functional.interpolate(x,scale_factor = 2**i, mode='bilinear')
            xs = self.conv_l_1(xs)
            xs = nn.functional.interpolate(xs,scale_factor = 0.5**i, mode = 'bilinear')
            E.append(xs)

        E = torch.stack(E)
        G_out = torch.reshape(G_out,(G_out.shape[1],G_out.shape[0],1,1,1))
        x = (E*G_out).sum(dim=0)

        x = self.conv2(x)
        return x.permute(0,2,3,1)


class Conv_All_Scale_Block(nn.Module):
    def __init__(self, D_features, r_features, kernel_size=3, act_layer=nn.GELU, num_experts = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(D_features,r_features,kernel_size = 1, padding = 'same', bias = True)
        self.convlist1 = nn.ModuleList()
        # self.conv_l2 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)
        
        for i in range(self.num_experts):
            conv_l = nn.Conv2d(r_features,
                              r_features,
                              kernel_size = kernel_size,
                              padding = 'same',
                              bias = True)
            self.convlist1.append(conv_l)

        
        # self.conv_l_1 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)
        # self.conv_l_2 = nn.Conv2d(r_features,
        #                       r_features,
        #                       kernel_size = kernel_size,
        #                       padding = 'same',
        #                       bias = True)

        self.conv2 = nn.Conv2d(r_features,D_features,kernel_size = 1, padding = 'same', bias = True)
        ## V2

        #initial_weights = torch.rand((self.num_experts,1,1,1,1),requires_grad = True)
        initial_weights = torch.ones((self.num_experts,1,1,1,1),requires_grad = True)
        # Normalize the initial weights so they sum to 1
        normalized_weights = initial_weights / initial_weights.sum()
        # Convert the normalized tensor to a learnable parameter
        self.w_a = torch.nn.parameter.Parameter(normalized_weights)

        #self.w_a = torch.nn.parameter.Parameter(torch.rand((self.num_experts,1,1,1,1),requires_grad = True))
        self.act = act_layer()
        #self.scale = torch.nn.parameter.Parameter(torch.tensor(1))
        self.scale = torch.nn.parameter.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_orig = x.permute(0,3,1,2).clone()
        # x : (B, window_size, window_size, embed_dim)
        x = self.conv1(x.permute(0,3,1,2)) # x shape (B,r,H,W)
        xe = x.clone()
        E = []
        for i in range(self.num_experts):
            
            xs = nn.functional.interpolate(x,scale_factor = 2**i, mode='bilinear')
            xs = self.convlist1[i](xs)
            xs = self.act(xs)
            xs = nn.functional.interpolate(xs,scale_factor = 0.5**i, mode = 'bilinear')
            E.append(xs)

        E = torch.stack(E)
        #x = E.sum(dim=0)
        # V2
        norm_w_a = F.softmax(self.w_a,dim=0)
        #self.w_a = self.w_a.softmax(0) #normalizing
        #x = (E*self.w_a).sum(dim=0)/self.w_a.sum(dim=0)
        x = (E*norm_w_a).sum(dim=0)/norm_w_a.sum(dim=0)
        x = x + self.scale*xe
        #x = self.conv_l2(x)
        #x = x + xe
        x = self.conv2(x)
        x = x + self.scale*x_orig
        return x.permute(0,2,3,1)

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
