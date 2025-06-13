import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )
        
        # 下采样路径
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        
        # 上采样路径
        self.up4 = upsample_block(1024, 512)
        self.up_conv4 = conv_block(1024, 512)
        
        self.up3 = upsample_block(512, 256)
        self.up_conv3 = conv_block(512, 256)
        
        self.up2 = upsample_block(256, 128)
        self.up_conv2 = conv_block(256, 128)
        
        self.up1 = upsample_block(128, 64)
        self.up_conv1 = conv_block(128, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        # 最大池化层
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # 下采样
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.maxpool(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.maxpool(conv4)
        
        conv5 = self.conv5(pool4)
        
        # 上采样
        up4 = self.up4(conv5)
        up4 = torch.cat([up4, conv4], dim=1)
        up_conv4 = self.up_conv4(up4)
        
        up3 = self.up3(up_conv4)
        up3 = torch.cat([up3, conv3], dim=1)
        up_conv3 = self.up_conv3(up3)
        
        up2 = self.up2(up_conv3)
        up2 = torch.cat([up2, conv2], dim=1)
        up_conv2 = self.up_conv2(up2)
        
        up1 = self.up1(up_conv2)
        up1 = torch.cat([up1, conv1], dim=1)
        up_conv1 = self.up_conv1(up1)
        
        out = self.final_conv(up_conv1)
        return torch.tanh(out)