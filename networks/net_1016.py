import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.nn.parameter import Parameter
import scipy.stats as st
import funcy
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

unloader = ToPILImage()
def imshow(tensor):
	image = unloader(tensor)
	plt.imshow(image)
	plt.pause(0.01)

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

class UNet11(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, has_dropout=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

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
        self.has_dropout = has_dropout


        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
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

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        final = self.final(dec1)

        return final


class UNet12(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, has_dropout=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder2 = models.vgg11(pretrained=pretrained).features

        self.relu2 = self.encoder2[1]
        self.conv12 = self.encoder2[0]
        self.conv22 = self.encoder2[3]
        self.conv3s2 = self.encoder2[6]
        self.conv32 = self.encoder2[8]
        self.conv4s2 = self.encoder2[11]
        self.conv42 = self.encoder2[13]
        self.conv5s2 = self.encoder2[16]
        self.conv52 = self.encoder2[18]
        self.dropout2 = nn.Dropout2d(p=0.3, inplace=False)
        self.has_dropout2 = has_dropout


        self.center2 = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec52 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec42 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec32 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec22 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec12 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final2 = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv12 = self.relu2(self.conv12(x))
        conv22 = self.relu2(self.conv22(self.pool2(conv12)))
        conv3s2 = self.relu2(self.conv3s2(self.pool2(conv22)))
        conv32 = self.relu2(self.conv32(conv3s2))
        conv4s2 = self.relu2(self.conv4s2(self.pool2(conv32)))
        conv42 = self.relu2(self.conv42(conv4s2))
        conv5s2 = self.relu2(self.conv5s2(self.pool2(conv42)))
        conv52 = self.relu2(self.conv52(conv5s2))
        if self.has_dropout2:
            conv52 = self.dropout2(conv52)

        center2 = self.center2(self.pool2(conv52))

        dec52 = self.dec52(torch.cat([center2, conv52], 1))
        dec42 = self.dec42(torch.cat([dec52, conv42], 1))
        dec32 = self.dec32(torch.cat([dec42, conv32], 1))
        dec22 = self.dec22(torch.cat([dec32, conv22], 1))
        dec12 = self.dec12(torch.cat([dec22, conv12], 1))
        final2 = self.final(dec12)

        return final2


def unet11(pretrained=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    model = UNet11(pretrained=pretrained, **kwargs)

    if pretrained == 'carvana':
        state = torch.load('TernausNet.pt')
        model.load_state_dict(state['model'])
    return model


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   # nn.BatchNorm2d(self.encoder[0].out_channels),
                                   self.relu,
                                   self.encoder[2],
                                   # nn.BatchNorm2d(self.encoder[2].out_channels),
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   # nn.BatchNorm2d(self.encoder[5].out_channels),
                                   self.relu,
                                   self.encoder[7],
                                   # nn.BatchNorm2d(self.encoder[7].out_channels),
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   # nn.BatchNorm2d(self.encoder[10].out_channels),
                                   self.relu,
                                   self.encoder[12],
                                   # nn.BatchNorm2d(self.encoder[12].out_channels),
                                   self.relu,
                                   self.encoder[14],
                                   # nn.BatchNorm2d(self.encoder[14].out_channels),
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   # nn.BatchNorm2d(self.encoder[17].out_channels),
                                   self.relu,
                                   self.encoder[19],
                                   # nn.BatchNorm2d(self.encoder[19].out_channels),
                                   self.relu,
                                   self.encoder[21],
                                   # nn.BatchNorm2d(self.encoder[21].out_channels),
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   # nn.BatchNorm2d(self.encoder[24].out_channels),
                                   self.relu,
                                   self.encoder[26],
                                   # nn.BatchNorm2d(self.encoder[26].out_channels),
                                   self.relu,
                                   self.encoder[28],
                                   # nn.BatchNorm2d(self.encoder[28].out_channels),
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            x_out = torch.sigmoid(x_out)

        return x_out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# concat block
class Soft_Att(nn.Module):
	def __init__(self):
		super(Soft_Att, self).__init__()
		self.Soft = Soft()
		self.pred = nn.Conv2d(64, 1,kernel_size=3, padding=1)  # concat [1, 8, 256, 256]
		self._initialize_weights()
		self.AlbuNet = AlbuNet()
		self.ra2_conv1 = BasicConv2d(33, 32, kernel_size=1)
		# self.ra2_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
		self.ra2_conv3 = BasicConv2d(32, 16, kernel_size=3, padding=1)
		self.ra2_conv4 = BasicConv2d(16, 1, kernel_size=3, padding=1)
		self.conv_block = nn.Sequential(
			nn.ConvTranspose2d(32, 16, 2, stride=2),
			nn.BatchNorm2d(16),
			nn.PReLU(),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.PReLU(),
			nn.Conv2d(16, 32, kernel_size=3,padding=1, stride=2),
			nn.BatchNorm2d(32),
			nn.PReLU()
		)



	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight.data, std=0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, output1, output2, uncer_output, feat1, feat2):   # ori_pic 为feature
		fea1 = self.conv_block(feat1) # [4, 1, 256, 256]
		fea2 = self.conv_block(feat2) # [4, 1, 256, 256]
		# print(fea.size())
		Att_out1 = torch.sigmoid(output1)
		Att_out2 = torch.sigmoid(output2)
		x1 = -1 * (torch.sigmoid(Att_out1)) + 1
		x2 = -1 * (torch.sigmoid(Att_out2)) + 1
		fea1 = torch.cat([fea1, uncer_output], dim=1)
		fea2 = torch.cat([fea2, uncer_output], dim=1)
		x1 = torch.mul(fea1, x1) # ([4, 33, 256, 256])
		x2 = torch.mul(fea2, x2)
		x1 = self.ra2_conv1(x1)
		# x1 = F.relu(self.ra2_conv2(x1))
		x1 = F.relu(self.ra2_conv3(x1))
		ra2_fea1 = self.ra2_conv4(x1)
		x2 = self.ra2_conv1(x2)
		# x2 = F.relu(self.ra2_conv2(x2))
		x2 = F.relu(self.ra2_conv3(x2))
		ra2_fea2 = self.ra2_conv4(x2)
		x2 = ra2_fea2 + Att_out2
		x1 = ra2_fea1 + Att_out1
		return [x1, x2]






# Soft Attention
def gkern(kernlen=16, nsig=3):
	interval = (2*nsig+1.)/kernlen
	x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel


def min_max_norm(in_):
	max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
	min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
	in_ = in_ - min_
	return in_.div(max_-min_+1e-8)

class Soft(nn.Module):
	# holistic attention module
	def __init__(self):
		super(Soft, self).__init__()
		gaussian_kernel = np.float32(gkern(31, 4))
		gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
		self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

	def forward(self, attention,x):
		soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
		soft_attention = min_max_norm(soft_attention)
		# soft_attention = soft_attention.max(attention)
		# print("max attention:", soft_attention)
		x = torch.mul(x, soft_attention.max(attention))
		return soft_attention


if __name__ == '__main__':
	# net =UNet11(pretrained=True)
	soft = Soft_Att()
	image1 = torch.randn(1,1,256,256)
	image2 = torch.randn(1,1,256,256)
	image3 = torch.randn(1,1,256,256)
	image4 = torch.randn(1,1,256,256)
	target1 = torch.randn(1,1,256,256)
	target2 = torch.randn(1,1,256,256)


	# out = soft(image1, image2, image3, image4, target1, target2)

	input = torch.randn(1, 9, 256, 256)
	Net = AlbuNet()
	output, fea = Net(input)
	print(fea.size())