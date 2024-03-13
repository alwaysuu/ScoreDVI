import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, affine=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.affine = affine

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.affine:
            return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        else:
            return (x - mu) / torch.sqrt(sigma+1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, affine=False):
        super(LayerNorm, self).__init__()
        
        self.body = WithBias_LayerNorm(dim, affine)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.body(x.permute(0, 2, 3, 1).reshape(b, -1, c))
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)

class Norm(nn.Module):
    def __init__(self, num_channel, norm_type='batchnorm'):
        super(Norm, self).__init__()
        
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm2d(num_channel, affine=True)
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm2d(num_channel, affine=False)
        elif norm_type == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=16, num_channels=num_channel, affine=False)
        elif norm_type == 'layernorm':
            self.norm = LayerNorm(dim=num_channel, affine=False)
        elif norm_type == 'none':
            self.norm = nn.Sequential()
        else:
            assert False
    
    def forward(self, x):
        return self.norm(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    
    def __init__(self, in_ch, out_ch, norm='batchnorm', kernel_size=3, padding=1):
        super(double_conv, self).__init__()
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
                Norm(out_ch, norm),
                nn.ReLU(inplace=True),
                )
            
    def forward(self, x):
        x = self.conv(x)

        return x
    
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm', kernel_size=3, padding=1):
        super(inconv, self).__init__()
        
        self.conv = double_conv(in_ch, out_ch, norm=norm, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)

        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm', kernel_size=3, padding=1):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bilinear=True, norm='batchnorm'):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2, 
                        diffX // 2, diffX - diffX // 2), 'replicate')
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, norm='batchnorm'):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch, norm)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x

class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, nIn=3, nOut=3, down_sample_norm='instancenorm', 
                 up_sample_norm = 'batchnorm', need_sigmoid=False, kernel_size=3, padding=1, bilinear=True):
        super(UNet, self).__init__()
        
        self.inc = inconv(nIn, 64, norm=down_sample_norm, kernel_size=kernel_size, padding=padding)
        self.down1 = down(64, 128, norm=down_sample_norm, kernel_size=kernel_size, padding=padding)
        self.down2 = down(128, 256, norm=down_sample_norm, kernel_size=kernel_size, padding=padding)
        self.down3 = down(256, 512, norm=down_sample_norm, kernel_size=kernel_size, padding=padding)
        self.down4 = down(512, 512, norm=down_sample_norm, kernel_size=kernel_size, padding=padding)
        self.up1 = up(1024, 256, norm=up_sample_norm, kernel_size=kernel_size, padding=padding, bilinear=bilinear)
        self.up2 = up(512, 128, norm=up_sample_norm, kernel_size=kernel_size, padding=padding, bilinear=bilinear)
        self.up3 = up(256, 64, norm=up_sample_norm, kernel_size=kernel_size, padding=padding, bilinear=bilinear)
        self.up4 = up(128, 64, norm=up_sample_norm, kernel_size=kernel_size, padding=padding, bilinear=bilinear)
        self.outc = outconv(64, nOut)
        
        self.need_sigmoid = need_sigmoid
        
    def forward(self, x):
        self.x1 = self.inc(x)        
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        self.x6 = self.up1(self.x5, self.x4)
        self.x7 = self.up2(self.x6, self.x3)
        self.x8 = self.up3(self.x7, self.x2)
        self.x9 = self.up4(self.x8, self.x1)     
        self.y = self.outc(self.x9)
        
        if self.need_sigmoid:
            self.y = torch.sigmoid(self.y)
        
        return self.y

    
class UNet_New(nn.Module):
    def __init__(self, nIn=3, nOut=3, down_sample_norm='instancenorm', 
                 up_sample_norm='batchnorm', kernel_size=3, padding=1, need_sigmoid=False, bilinear=True):
        super(UNet_New, self).__init__()
        
        self.net = UNet(nIn, nOut, down_sample_norm, up_sample_norm, need_sigmoid, kernel_size, padding, bilinear)
        
    def forward(self, x):
        x = self.net(x)
        
        return x

