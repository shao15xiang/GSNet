# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:32:56 2019

@author: Shao Xiang, Hunan university, email: xs15@hnu.edu.cn
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size = 1, stride = 1, padding=0,groups = 1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,groups=groups, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x



class GS(nn.Module):
    def __init__(self,inc,outc, s = 1,reduction = 2, downsample = None,dp = 0):
        super(GS,self).__init__()
        self.inc = inc
        self.outc = outc
        self.downsample = downsample
        self.drop = dp
        self.conv0 = nn.Sequential(nn.Conv2d(self.inc ,self.inc // reduction, 
                               kernel_size = 1),
                                nn.BatchNorm2d(self.inc // reduction),
                                nn.ReLU6(inplace = True))
        
        self.conv1 = nn.Conv2d(self.inc // (reduction*4),self.inc // (reduction*4) , 
                               kernel_size = 3,
                               stride = 1,
                                padding= 1,
                               groups = 1)
        
        self.bn1 = nn.BatchNorm2d(self.inc //(reduction*4))
        self.relu = nn.ReLU6(inplace = True)
        self.conv2 = nn.Conv2d(self.inc // (reduction*2),self.inc // (reduction*2), 
                               kernel_size = 3,stride = 1, padding= 1,
                               groups = 1)
        
        self.bn2 = nn.BatchNorm2d(self.inc // (reduction*2))
        self.conv3 = nn.Sequential(nn.Conv2d(3*self.inc // (reduction*4) ,3*self.inc // (reduction*4) , 
                               kernel_size = 3,stride = 1, padding= 1,groups = 1),
                                nn.BatchNorm2d(3*self.inc // (reduction*4)),
                                nn.ReLU6(inplace = True))
        
        self.pool = nn.MaxPool2d(3,1,1)
#        self.se = SEModule(self.outc,16)
        if self.downsample:
            self.base_conv = BasicConv2d(7*self.inc // (reduction*4),self.outc,3,s,1)
        else:
            self.base_conv = BasicConv2d(7*self.inc // (reduction*4),self.outc,groups = 1)
        
        
    def forward(self,x):
        res = x
        x = self.conv0(x)
        inc = x.size(1) // 4
        split_x = x.split(inc,1)
        
        x = self.conv1(split_x[0])
        x1 = self.relu(self.bn1(x))   
        
        x = torch.cat((x1,split_x[1]),1)
        
        x = channel_shuffle(x,2)
        x = self.conv2(x)
        x2 = self.relu(self.bn2(x))
        
        
        x = torch.cat((x2,split_x[2]),1)        
        x = channel_shuffle(x,2)
        x3 = self.conv3(x)
        
        
        x4 = self.pool(split_x[3])
        x = torch.cat((x1,x2,x3,x4),1)  
        
        x = channel_shuffle(x,2)
#        print(x.size())
        x = self.base_conv(x) 
        
        if self.drop > 0:
            x = F.dropout(x, p = self.drop, training=self.training)
        
        if self.downsample:
            res = self.downsample(res)
#        x = self.se(x)
        x = res+ x
        x = self.relu(x)
        return x
    
    

       
class GSNet(nn.Module):
    def __init__(self,layer,num_class = 10,expansion = 2,dropout = 0):
        super(GSNet,self).__init__()
        
        assert expansion in [1,2,3,4,5,6,7,8,9], print('The vaule of expansion is error, which range is [1,2,3,4]')
        if expansion == 1:
            N = [16,32,48,64]
            
        elif expansion == 2:
            N = [32,48,64,96]
            
        elif expansion == 3:
            N = [32,64,128,256]
            
        elif expansion == 4:
            N = [64,128,256,512]
            
        elif expansion == 5:
            N = [64,256,384,968]
            
        elif expansion == 6:
            N = [64,512,768,1536]
            
        elif expansion == 7:
            N= [64*4,128*4,256*4,512*4]
            
        elif expansion == 8:
            N = [64*5,128*5,256*5,512*5]
            
        else:
            N = [64*5,128*5,256*6,512*6]
            
            
        self.expansion = expansion 
        self.inplane = 64
        self.conv = BasicConv2d(3,N[0],7,2,3) 
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.dropout = dropout
        self.reduction = 2
        
        self.layer1 = self.make_layer(layer[0],N[0],N[0])
        self.layer2 = self.make_layer(layer[1],N[0],N[1])
        self.layer3 = self.make_layer(layer[2],N[1],N[2])
                
        self.layer4 = nn.Sequential(                
                nn.Conv2d(N[2],N[3],1,1,0,bias = True),
                nn.BatchNorm2d(N[3]),
                nn.ReLU(inplace = True))
                
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc =  nn.Linear(N[3],num_class)   
                
        self.initialize_weights()
                
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
    def make_layer(self,layer,inc,outc,s = 2,ds = None):
        blocks = []
        
        if s == 2 or inc != outc:    
            ds = nn.Sequential(nn.Conv2d(inc,outc,3,s,1),
                                       nn.BatchNorm2d(outc),
                                       nn.ReLU(inplace = True))
        
        blocks.append(GS(inc, outc, s,  reduction = self.reduction,
                             downsample = ds,dp = self.dropout))
        self.inplane = outc
        
        for i in range(1,layer):
            s = 1
            blocks.append(GS(self.inplane, outc, s, reduction= self.reduction, downsample = None,dp = self.dropout))       
        return nn.Sequential(*blocks)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x =self.avg(x)
        x = x.view(x.size(0),-1)
        x = F.dropout(x, p = 0.1, training=self.training)
        x = self.fc(x)                   
        return x
    
if __name__=='__main__':
    
    net = GSNet([8,7,3],100,7)
    input = torch.randn(1,3,64,64)
    
    output = net(input)
    #print(output)
    params = list(net.parameters())
    num = 0
    for i in params:
        l=1
        #print('Size:{}'.format(list(i.size())))
        for j in i.size():
            l *= j
        num += l
    print('All Parameters:{}'.format(num))

        
        
