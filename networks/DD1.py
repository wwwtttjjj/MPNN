import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch import Tensor
from networks.layers import *
import itertools
from torch.distributions.uniform import Uniform
import numpy as np

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)  # 直接对某层的data进行复制
        m.bias.data.fill_(0)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ShareEncoder(nn.Module):
    def __init__(self, nin, pretrained=True,has_dropout=False):
        # 把unet11的拿过来了
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.has_dropout = has_dropout
        # self.encoder = models.vgg11(pretrained=pretrained).features

        self.encoder = models.vgg11(pretrained=pretrained).features
        # self.encoder = self.add_dropout(pretrained)

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.dropout = nn.Dropout2d(p=0.3, inplace=False)

    def forward(self, x):
        x = x.float()
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))
        if self.has_dropout:
            conv5 = self.dropout(conv5)
        return conv5, conv1, conv2, conv3, conv4

class Decoder1(nn.Module):
    def __init__(self,num_filters=32,nout=2):
        super().__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.center1 = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec51 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec41 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec31 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec21 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec11 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final1 = nn.Conv2d(num_filters, nout, kernel_size=1) # 2通道出去

    def forward(self, conv5, conv1, conv2, conv3, conv4):
        center1 = self.center1(self.pool1(conv5))
        dec51 = self.dec51(torch.cat([center1, conv5], 1))
        dec41 = self.dec41(torch.cat([dec51, conv4], 1))
        dec31 = self.dec31(torch.cat([dec41, conv3], 1))
        dec21 = self.dec21(torch.cat([dec31, conv2], 1))
        dec11 = self.dec11(torch.cat([dec21, conv1], 1))
        final1 = self.final1(dec11)

        return final1, dec11

class Decoder2(nn.Module):
    def __init__(self,num_filters=32,nout=2):
        super().__init__()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.center2 = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec52 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec42 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec32 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec22 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec12 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final2 = nn.Conv2d(num_filters, nout, kernel_size=1) # 2通道出去

    def forward(self, conv5, conv1, conv2, conv3, conv4):
        center2 = self.center2(self.pool2(conv5))
        dec52 = self.dec52(torch.cat([center2, conv5], 1))
        dec42 = self.dec42(torch.cat([dec52, conv4], 1))
        dec32 = self.dec32(torch.cat([dec42, conv3], 1))
        dec22 = self.dec22(torch.cat([dec32, conv2], 1))
        dec12 = self.dec12(torch.cat([dec22, conv1], 1))
        final2 = self.final2(dec12)

        return final2, dec12

class UnFNet1(nn.Module):
    def __init__(self, nin, nout, device, l_rate, pretrained=True, has_dropout=True, num_filters=32):
        super().__init__()
        self.encoder = ShareEncoder(nin, pretrained=pretrained, has_dropout=has_dropout).to(device)
        self.decoder1 = Decoder1(num_filters=num_filters, nout=nout).to(device)
        self.decoder2 = Decoder2(num_filters=num_filters, nout=nout).to(device)

        # self.encoder.apply(weights_init)
        self.decoder1.apply(weights_init)
        self.decoder2.apply(weights_init)

        optimizer01 = torch.optim.Adam(self.encoder.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer11 = torch.optim.Adam(self.decoder1.parameters(), lr=l_rate, betas=(0.9, 0.99))
        optimizer21 = torch.optim.Adam(self.decoder2.parameters(), lr=l_rate, betas=(0.9, 0.99))

        self.optimizers = [optimizer01, optimizer11, optimizer21]
    def forward(self, input):
        conv5, conv1, conv2, conv3, conv4 = self.encoder(input)
        # print(conv1.size(), conv2.size(), conv3.size(), conv4.size(), conv5.size())
        res1, fea1 = self.decoder1(conv5, conv1, conv2, conv3, conv4)
        res2, fea2 = self.decoder2(conv5, conv1, conv2, conv3, conv4)
        return [res1, res2]

    def optimize(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()



if __name__ == '__main__':
    device = torch.device("cuda")
    lr = 0.01
    input = torch.randn((4,3,256,256)).to(device)
    # model = UnFNet(3, 2, device, lr, pretrained=True, has_dropout=True)
    # # print(model)
    # res = model(input)
    # for i in range(5):
    #     out = model(input)
    # print(out[0][0].size())

    net = UNet_URPC(in_chns=3, class_num=1).cuda()
    res = net(input)
    # print(res[2].size())



