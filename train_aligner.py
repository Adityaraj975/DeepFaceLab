import multiprocessing
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from core import osex
from core.cv2ex import *
from train_aligner_generator import SampleGenerator

if __name__ == "__main__":
    
    osex.set_process_lowest_prio()
    multiprocessing.set_start_method("spawn")

    resolution = 224
    batch_size = 64

    generator = SampleGenerator(r'F:\DeepFaceLabCUDA9.2SSE\wf_faces', batch_size=batch_size, resolution=resolution, generators_count=6)#debug=True, 

    class FaceAligner(nn.Module):
        def __init__(self, resolution):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
            self.conv3 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
            self.conv4 = nn.Conv2d(128, 256, 5, stride=1, padding=2)
            self.conv5 = nn.Conv2d(256, 512, 5, stride=1, padding=2)
            low_res = resolution // (2**5)
                        
            self.fc1 = nn.Linear(512*low_res*low_res, 6)
            
        def forward(self, x):
            
            x = F.leaky_relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.leaky_relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.leaky_relu(self.conv3(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.leaky_relu(self.conv4(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.leaky_relu(self.conv5(x))
            x = F.max_pool2d(x, 2, 2)
            
            x = x.reshape( x.shape[0], -1 )
            
            x = self.fc1(x)
            return x
            
            import code
            code.interact(local=dict(globals(), **locals()))
            
            return x
            
    #net = FaceAligner(resolution)        
    net = tv.models.resnet.ResNet ( tv.models.resnet.BasicBlock, [2,2,2,2], 6)
    #net = tv.models.mobilenet_v2 ( num_classes = 6)
    #net = tv.models.resnext50_32x4d ( num_classes = 6)
    #net = tv.models.inception_v3 ( num_classes = 6)
    net.cuda()
    net.train()
    

    opt = torch.optim.Adam(net.parameters(), lr=5e-7 )
    
    iter = 0
    
    model_ckp_path = Path(r'E:\face_aligner.pt')
    if model_ckp_path.exists():
        checkpoint = torch.load(model_ckp_path)    
        iter = checkpoint['iter']
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # if iter == 0:
    #     for m in net.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             nn.init.zeros_(m.bias)
    #             #nn.init.zeros_(m.weight)
    #             #m.bias.data = torch.from_numpy ( np.array([1,0,0,0,1,0], np.float32) ).cuda()
        
    # img = cv2_imread(r'D:\DevelopPython\test\00006.jpg')
    # img = img.transpose( (2,0,1) )[None,...].astype(np.float32)
    # img /= 255.0
    # pred_mats_t = net( torch.from_numpy(img).cuda() )
    # #pred_mats_t = pred_mats_t.detach().cpu().numpy().reshape( (2,3) )
    # #pred_mats_t[:,2] *= resolution
    # print( pred_mats_t )
    # import code        
    # code.interact(local=dict(globals(), **locals())) 
    
    while True:
            
        imgs, mats = next(generator)
        
        #for img in imgs:
        #    cv2.imshow("", img.transpose( (1,2,0) ) )
        #    cv2.waitKey(0)
        # mat = mats[0].reshape( (2,3))
        # mat[:,2] *= 224
        # import code
        # code.interact(local=dict(globals(), **locals()))
        
        
        
        # cv2.imshow("",  cv2.warpAffine(img, mat, (224, 224), cv2.INTER_LANCZOS4, flags=cv2.WARP_INVERSE_MAP) )
        # cv2.waitKey(0)
        
        imgs_t = torch.from_numpy(imgs).cuda()
        mats_t = torch.from_numpy(mats).cuda()
        
        opt.zero_grad()
        
        pred_mats_t = net(imgs_t)
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        loss = (pred_mats_t-mats_t).square().mean()
        loss.backward()
        
        opt.step()
        #lr_sched.step()
        
        print(f'iter = {iter} loss = {loss.detach().cpu().numpy()} ')
        
        if iter % 100 == 0:
            print('Saving')
            torch.save({'iter':iter,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        }, model_ckp_path )
        iter += 1
        #import code
        #code.interact(local=dict(globals(), **locals()))
