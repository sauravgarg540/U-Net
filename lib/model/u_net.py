import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv_block(x)

class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, features = [64,128,256,512]):
        super(UNet, self).__init__()

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.features = features

        for feature in self.features:
            self.down.append(ConvBlock(in_channels, feature))
            in_channels = feature
        
        for feature in self.features[::-1]:
            self.up.append(nn.ConvTranspose2d(2*feature, feature, 2, stride=2))
            self.up.append(ConvBlock(2*feature, feature))
        self.bottleneck = ConvBlock(self.features[-1], 2*self.features[-1])
        self.final_layer = nn.Conv2d(self.features[0], out_channels,1 ,padding = 0)

    def forward(self, x):
        encoded = []
        for i in range(len(self.features)):
            x = self.down[i](x)
            encoded.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)
        encoded = encoded[::-1]
        for i in range(0,len(self.up),2):
            x = self.up[i](x)
            _,_,H,W = x.shape
            x_down = tf.center_crop(encoded[i//2],[H,W])
            x = torch.cat([x,x_down], dim = 1)
            x = self.up[i+1](x)
        x = self.final_layer(x)
        return x


# test model
# if __name__ == "__main__":
#     model = UNet(1, 2)
#     x = torch.randn([1,1,572,572])
#     model(x)

        
    