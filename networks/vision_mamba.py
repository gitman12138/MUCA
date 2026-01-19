# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM

logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    def __init__(self, config=None, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.mamba_unet = VSSM(num_classes=self.num_classes)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


model = MambaUnet(img_size=512,num_classes=7)
#
# dummy_input = torch.randn(1, 1, 256, 256)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))



# 假设你已经加载模型 model，并设置为 eval 模式
model.eval()
model.cuda()

# 模拟一张输入图像
input_tensor = torch.randn(1, 1, 512, 512).cuda()  # (B, C, H, W)

# GPU 预热
for _ in range(10):
    _ = model(input_tensor)

torch.cuda.synchronize()
start_time = time.time()

num_runs = 100
for _ in range(num_runs):
    _ = model(input_tensor)
torch.cuda.synchronize()  # 等待所有 GPU 操作完成

end_time = time.time()
avg_inference_time = (end_time - start_time) / num_runs * 1000  # 毫秒单位

print(f"Average inference time per image: {avg_inference_time:.2f} ms")
