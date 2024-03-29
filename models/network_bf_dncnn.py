import torch
import torch.nn as nn


class BF_DNCNN(nn.Module):

    def __init__(self, num_kernels=64, num_layers=17, num_channels_in=3, num_channels_out=3):
        super(BF_DNCNN, self).__init__()

        self.padding = 1
        self.num_kernels = num_kernels
        self.kernel_size = 3
        self.num_layers = num_layers
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out

        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])

        self.relu = nn.ReLU(inplace=True)

        self.conv_layers.append(nn.Conv2d(self.num_channels_in,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))
            self.running_sd.append( nn.Parameter(torch.ones(1,self.num_kernels,1,1), requires_grad=False) )
            g = (torch.randn( (1,self.num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True) )

        self.conv_layers.append(nn.Conv2d(self.num_kernels,self.num_channels_out, self.kernel_size, padding=self.padding , bias=False))


    def forward(self, x, condition=None):
        raw_x = x
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
            
        x = self.relu(self.conv_layers[0](x))
        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            # BF_BatchNorm
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)

            if self.conv_layers[l].training:
                x = x / sd_x.expand_as(x)
                self.running_sd[l-1].data = (1-.1) * self.running_sd[l-1].data + .1 * sd_x
                x = x * self.gammas[l-1].expand_as(x)

            else:
                x = x / self.running_sd[l-1].expand_as(x)
                x = x * self.gammas[l-1].expand_as(x)

            x = self.relu(x)

        x = self.conv_layers[-1](x)
        
        return raw_x - x

        '''if self.num_channels_in == self.num_channels_out:
            return raw_x - x
        else:
            return raw_x - x[:, :self.num_channels_in, ...], x[:, self.num_channels_in:, ...]'''