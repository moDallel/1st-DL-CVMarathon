# %%
from ssd import build_ssd
from layers.box_utils import *
import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import torchvision
import pickle
from layers import box_utils
from layers import Detect
from layers import functions
from layers import modules
import torch.nn.functional as F
from math import sqrt as sqrt
from itertools import product as product

from torch.autograd import Function
from layers.box_utils import decode, nms

# %%
## 詳細模型結構可以參考ssd.py
ssd_net=build_ssd('train', size=300, num_classes=21)
ssd_net.load_weights('./demo/ssd300_mAP_77.43_v2.pth')

# %%
'''
## 默認Config檔案在data/config.py內
'''

# %%
ssd_net.cfg

# %%
cfg = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# %%
'''
### 'aspect_ratios' : 使用六張Feature Map，每一張上方有預設的anchor boxes，Boxes aspect ratio可以自己設定
### 'feature_maps' : 使用feature map大小為[38x38, 19x19, 10x10, 5x5, 3x3, 1x1]
### 'min_sizes'、'max_sizes'可藉由下方算式算出，由作者自行設計
### 'steps' : Feature map回放回原本300*300的比例，如38要回放為300大概就是8倍
### 'variance' : Training 的一個trick，加速收斂，詳見：https://github.com/rykov8/ssd_keras/issues/53
'''

# %%
'''
---
'''

# %%
'''
## 'min_sizes'、'max_sizes' 計算
'''

# %%
import math
## source:https://blog.csdn.net/gbyy42299/article/details/81235891
min_dim = 300   ## 维度
# conv4_3 ==> 38 x 38
# fc7 ==> 19 x 19
# conv6_2 ==> 10 x 10
# conv7_2 ==> 5 x 5
# conv8_2 ==> 3 x 3
# conv9_2 ==> 1 x 1
mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2'] ## prior_box來源層，可以更改。很多改進都是基於此處的調整。
# in percent %
min_ratio = 20 ## 這裡即是論文中所說的Smin的= 0.2，Smax的= 0.9的初始值，經過下面的運算即可得到min_sizes，max_sizes。
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))## 取一個間距步長，即在下面用於循環給比取值時起一個間距作用。可以用一個具體的數值代替，這裡等於17。
min_sizes = []  ## 經過以下運算得到min_sizes和max_sizes。
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
    ## 從min_ratio至max_ratio + 1每隔步驟= 17取一個值賦值給比。注意範圍函數的作用。
    ## min_sizes.append（）函數即把括號內部每次得到的值依次給了min_sizes。
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes

## steps: 這一步要仔細理解，即計算卷積層產生的prior_box距離原圖的步長，先驗框中心點的坐標會乘以step，
## 相當於從特徵映射位置映射回原圖位置，比如conv4_3輸出特徵圖大小為38 *38，而輸入的圖片為300* 300，
## 所以38 *8約等於300，所以映射步長為8.這是針對300* 300的訓練圖片。
steps = [8, 16, 32, 64, 100, 300]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

print('min_sizes: ',min_sizes)
print('max_sizes: ',max_sizes)


# %%
'''
---
'''

# %%
'''
## Default anchor boxes設計原理，看懂收穫很多
##### 可以理解 SSD原文中 8732個anchors是怎麼來的
##### 38×38×4+19×19×6+10×10×6+5×5×6+3×3×4+1×1×4=8732
'''

# %%
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        '''依照Feature map大小找出所有的pixel 中心'''
        '''下方這兩個loop會找出W個x軸pixel對上W個y軸pixel，假如現在是在38x38的feature map上，就會有38x38個值'''
        '''ex. [0,1],[0,2]..[0,37] [1,1],[1,2]..[1,37]..........[37,37]'''
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k] ## 如self.steps==8，就是先將原圖size normalize(/300)後再乘上8
                # unit center x,y
                '''中心點'''
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                '''/self.image_size 就是在做normalization '''
                s_k = self.min_sizes[k]/self.image_size
                '''小的正方形box'''
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                '''大的正方形box'''
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    '''aspect ratio 2,3'''
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    '''aspect ratio 1/2,1/3'''
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

# %%
PriorBox_Demo=PriorBox(cfg)

# %%
print(PriorBox_Demo.forward().shape)

# %%
'''
---
'''

# %%
'''
## Loss 如何設計
'''

# %%
class MultiBoxLoss(nn.Module):

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        '''有幾類'''
        self.num_classes = num_classes
        '''判定為正樣本的threshold，一般設為0.5'''
        self.threshold = overlap_thresh
        '''background自己會有一類，不用Label，假如我們有20類一樣標註0-19，下方會自己空出一類給background'''
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        '''OHEM，找出分得最不好的樣品，也就是confidence score比較低的正負樣品'''
        self.do_neg_mining = neg_mining
        '''負樣品與正樣品的比例，通常是3:1'''
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']


    def forward(self, predictions, targets):

        '''prediction會output三個值'''
        '''loc shape: bounding box 資訊，torch.size(batch_size,num_priors,4)'''
        '''conf shape: 每一個bounding box 的信心程度，torch.size(batch_size,num_priors,num_classes)'''
        '''priors shape: 預設的defaul box， torch.size(num_priors,4)'''
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            '''jaccard 計算每一個BBOX與ground truth的IOU'''
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        '''用Variable包裝'''
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)


        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        '''smooth_l1_loss 計算bounding box regression'''
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        '''排列confidence 的分數'''
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        '''負樣品取出數量 == negpos_ratio*num_pos'''
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        '''用cross_entropy做分類'''
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        #double轉成torch.float64
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


# %%
'''
## 產生我們Loss function，注意這裡的class要包含背景
'''

# %%
Use_cuda=False
criterion = MultiBoxLoss(21, 0.5, True, 0, False, 3, 0.5,False, Use_cuda,)

# %%
'''
----
'''

# %%
'''
## 基本設定
'''

# %%
ssd_net=build_ssd('train', size=300, num_classes=21)
use_pretrained=True
if use_pretrained:
    ssd_net.load_weights('./demo/ssd300_mAP_77.43_v2.pth')
net=ssd_net

# %%
'''要不要使用gpu'''
Use_cuda=False

'''tensor type會依照cpu或gpu有所不同'''
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

'''使用GPU時可以開啟DataParallel，但當Input是不定大小時，要關掉'''
if Use_cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
'''使用GPU時模型要轉成cuda'''
if Use_cuda:
    net = net.cuda()

batch_size_=1
optimizer = optim.Adam(net.parameters(),lr=0.00001/batch_size_)

# %%
'''
---
'''

# %%
'''
## 訓練
'''

# %%
'''
## 這裡我們先示範輸入的 image,Label格式，真正在訓練時，準備成一樣格式即可
'''

# %%
'''輸入影像格式，假設batch size 為 4'''
image_in=torch.tensor(torch.rand(4,3,300,300),dtype=torch.float32)
'''Label格式，沒有固定長度，看圖像中有幾個label就有幾個'''
label_0=[[ 0.1804,  0.6076,  0.7701,  0.8485, 0.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 3.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 19.0000],
       [ 0.2950,  0.0000,  0.8238,  0.3641, 6.0000],]
label_1=[[ 0.1804,  0.6076,  0.7701,  0.8485, 13.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 11.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 7.0000],
       [ 0.2950,  0.0000,  0.8238,  0.3641, 5.0000],]
label_2=[[ 0.1804,  0.6076,  0.7701,  0.8485, 0.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 3.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 14.0000],
       [ 0.2950,  0.0000,  0.8238,  0.3641, 6.0000],]
label_3=[[ 0.1804,  0.6076,  0.7701,  0.8485, 0.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 3.0000],
       [ 0.2250,  0.0000,  0.9238,  0.5641, 19.0000],
       [ 0.2950,  0.0000,  0.8238,  0.3641, 6.0000],]

# %%
#epochs=300
#iteration=1000
epochs=2
iteration=100

# %%
for epoch in range(epochs):
    n=0
    loss_sum=[]
    loc_loss=[]
    conf_loss=[]
    for number__ in range(iteration) :
        '''要用Variable包裝tensor才能送入模型'''
        if Use_cuda:
            image_ = Variable(image_in.cuda())
            y = [Variable(torch.tensor(label_0).cuda(), volatile=True),Variable(torch.tensor(label_1).cuda(),
                volatile=True),Variable(torch.tensor(label_2).cuda(), volatile=True),Variable(torch.tensor(label_3).cuda(), volatile=True)]
        else:
            image_ = Variable(image_in)
            y = [Variable(torch.tensor(label_0), volatile=True),Variable(torch.tensor(label_1),
                volatile=True),Variable(torch.tensor(label_2), volatile=True),Variable(torch.tensor(label_3), volatile=True)]

        '''Forward Pass'''
        out = net(image_)
        '''Regression Loss and Classification Loss'''
        loss_l,loss_c = criterion(out,y )
        loss = loss_l+ loss_c
        '''Backward'''
        loss.backward()

        loc_loss.append(loss_l.data.cpu().numpy())
        conf_loss.append(loss_c.data.cpu().numpy())
        loss_sum.append(loss.data.cpu().numpy())
        '''更新參數'''
        optimizer.step()
        '''清空Gradients'''
        optimizer.zero_grad()

        n+=1
        if n%10==0:
            print('BBOX Regression Loss: ', np.mean(loc_loss))
            print('Classification Loss: ', np.mean(conf_loss))
    '''儲存權重'''
    torch.save(ssd_net.state_dict(),'weights/Ｗeights.pth')

# %%
