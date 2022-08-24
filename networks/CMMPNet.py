import torch
from torchvision import models
from .basic_blocks import *
import math
import torch.nn.functional as F


class SPPLayer(torch.nn.Module):
    def __init__(self, block_size=[1,2,4], pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.block_size = block_size
        self.pool_type = pool_type
        self.spp = self.make_spp(out_pool_size=self.block_size, pool_type=self.pool_type)

    def make_spp(self, out_pool_size, pool_type='maxpool'):
        func=[]
        for i in range(len(out_pool_size)):
            if pool_type == 'max_pool':
                func.append(nn.AdaptiveMaxPool2d(output_size=(out_pool_size[i],out_pool_size[i])))
            if pool_type == 'avg_pool':
                func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i],out_pool_size[i])))
        return func

    def forward(self, x):
        num = x.size(0)
        for i in range(len(self.block_size)):
            tensor = self.spp[i](x).view(num, -1)
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten
        
        
        
class DEM(torch.nn.Module): # Dual Enhancement Module
    def __init__(self, channel, block_size=[1,2,4]):
        super(DEM, self).__init__()

        self.rgb_local_message = self.local_message_prepare(channel, 1, 1, 0)
        self.add_local_message = self.local_message_prepare(channel, 1, 1, 0)
        
        self.rgb_spp = SPPLayer(block_size=block_size)
        self.add_spp = SPPLayer(block_size=block_size)
        self.rgb_global_message = self.global_message_prepare(block_size, channel)
        self.add_global_message = self.global_message_prepare(block_size, channel)
        
        self.rgb_local_gate  = self.gate_build(channel*2, channel, 1, 1, 0)
        self.rgb_global_gate = self.gate_build(channel*2, channel, 1, 1, 0)
        
        self.add_local_gate  = self.gate_build(channel*2, channel, 1, 1, 0)
        self.add_global_gate = self.gate_build(channel*2, channel, 1, 1, 0)
        
        
    def local_message_prepare(self, dim, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding),  
            nn.BatchNorm2d(dim)
        )

    def global_message_prepare(self, block_size, dim):
        num_block = 0
        for i in block_size:
            num_block += i*i
        
        return  nn.Sequential(
            nn.Linear(num_block*dim, dim),
            nn.ReLU()
        )
        
    def gate_build(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding), 
            nn.Sigmoid()
        )
                
    def forward(self, rgb_info, add_info):
        rgb_local_info = self.rgb_local_message(rgb_info)
        add_local_info = self.add_local_message(add_info)
        
        rgb_global_info = torch.unsqueeze(torch.unsqueeze(self.rgb_global_message(self.rgb_spp(rgb_local_info)),-1),-1).expand(rgb_local_info.size())
        add_global_info = torch.unsqueeze(torch.unsqueeze(self.add_global_message(self.add_spp(add_local_info)),-1),-1).expand(add_local_info.size())
                
        rgb_info = rgb_info + add_local_info * self.add_local_gate(torch.cat((add_local_info, add_global_info), 1)) + add_global_info * self.add_global_gate(torch.cat((add_local_info, add_global_info), 1))
        add_info = add_info + rgb_local_info * self.rgb_local_gate(torch.cat((rgb_local_info, rgb_global_info), 1)) + rgb_global_info * self.rgb_global_gate(torch.cat((rgb_local_info, rgb_global_info), 1))

        return rgb_info, add_info        


class DinkNet34_CMMPNet(nn.Module):
    def __init__(self, block_size='1,2,4'):
        super(DinkNet34_CMMPNet, self).__init__()
        filters = [64, 128, 256, 512]
        self.net_name = "CMMPnet"       
        self.block_size = [int(s) for s in block_size.split(',')]

        # img
        resnet = models.resnet34(pretrained=True)
        self.firstconv1 = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DBlock(filters[3])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], filters[0]//2, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(filters[0]//2, filters[0]//2, 3, padding=1)
        self.finalrelu2 = nonlinearity


        ## addinfo, e.g, gps_map, lidar_map
        resnet1 = models.resnet34(pretrained=True)
        self.firstconv1_add = nn.Conv2d(1, filters[0], kernel_size=7, stride=2, padding=3)
        self.firstbn_add = resnet1.bn1
        self.firstrelu_add = resnet1.relu
        self.firstmaxpool_add = resnet1.maxpool
        
        self.encoder1_add = resnet1.layer1
        self.encoder2_add = resnet1.layer2
        self.encoder3_add = resnet1.layer3
        self.encoder4_add = resnet1.layer4

        self.dblock_add = DBlock(filters[3])

        self.decoder4_add = DecoderBlock(filters[3], filters[2])
        self.decoder3_add = DecoderBlock(filters[2], filters[1])
        self.decoder2_add = DecoderBlock(filters[1], filters[0])
        self.decoder1_add = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1_add = nn.ConvTranspose2d(filters[0], filters[0]//2, 4, 2, 1)
        self.finalrelu1_add = nonlinearity
        self.finalconv2_add = nn.Conv2d(filters[0]//2, filters[0]//2, 3, padding=1)
        self.finalrelu2_add = nonlinearity


        ### DEM
        self.dem_e1 = DEM(filters[0], self.block_size)
        self.dem_e2 = DEM(filters[1], self.block_size)
        self.dem_e3 = DEM(filters[2], self.block_size)
        self.dem_e4 = DEM(filters[3], self.block_size)

        self.dem_d4 = DEM(filters[2], self.block_size)        
        self.dem_d3 = DEM(filters[1], self.block_size)
        self.dem_d2 = DEM(filters[0], self.block_size)
        self.dem_d1 = DEM(filters[0], self.block_size)    
                    
                    
        self.finalconv = nn.Conv2d(filters[0], 1, 3, padding=1)


    def forward(self, inputs):

        x   = inputs[:,:3,:,:]    # image
        add = inputs[:,3:,:,:]    # gps_map or lidar_map

        x   = self.firstconv1(x)
        add = self.firstconv1_add(add)
        x   = self.firstmaxpool(self.firstrelu(self.firstbn(x)))
        add = self.firstmaxpool_add(self.firstrelu_add(self.firstbn_add(add)))

        x_e1   = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1, add_e1 = self.dem_e1(x_e1, add_e1)
        
        x_e2   = self.encoder2(x_e1)
        add_e2 = self.encoder2_add(add_e1)
        x_e2, add_e2 = self.dem_e2(x_e2, add_e2)
        
        x_e3   = self.encoder3(x_e2)
        add_e3 = self.encoder3_add(add_e2)
        x_e3, add_e3 = self.dem_e3(x_e3, add_e3)
 
        x_e4   = self.encoder4(x_e3)
        add_e4 = self.encoder4_add(add_e3)
        x_e4, add_e4 = self.dem_e4(x_e4, add_e4)
        
        # Center
        x_c   = self.dblock(x_e4)
        add_c = self.dblock_add(add_e4)

        # Decoder
        x_d4   = self.decoder4(x_c) + x_e3
        add_d4 = self.decoder4_add(add_c) + add_e3
        x_d4, add_d4 = self.dem_d4(x_d4, add_d4)
        
        x_d3   = self.decoder3(x_d4) + x_e2
        add_d3 = self.decoder3_add(add_d4) + add_e2
        x_d3, add_d3 = self.dem_d3(x_d3, add_d3)        
        
        x_d2   = self.decoder2(x_d3) + x_e1
        add_d2 = self.decoder2_add(add_d3) + add_e1
        x_d2, add_d2 = self.dem_d2(x_d2, add_d2)
        
        x_d1   = self.decoder1(x_d2)
        add_d1 = self.decoder1_add(add_d2)
        x_d1, add_d1 = self.dem_d1(x_d1, add_d1)
        
        x_out   = self.finalrelu1(self.finaldeconv1(x_d1))
        add_out = self.finalrelu1_add(self.finaldeconv1_add(add_d1))
        x_out   = self.finalrelu2(self.finalconv2(x_out))
        add_out = self.finalrelu2_add(self.finalconv2_add(add_out))

        out = self.finalconv(torch.cat((x_out, add_out),1)) # b*1*h*w
        return torch.sigmoid(out)
        
        