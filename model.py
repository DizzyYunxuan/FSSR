from torch import nn
import torch
import functools
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class Generator(nn.Module):
    def __init__(self, n_res_blocks=8):
        super(Generator, self).__init__()
        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        block = self.block_input(x)
        for res_block in self.res_blocks:
            block = res_block(block)
        block = self.block_output(block)
        return torch.sigmoid(block)

class De_resnet(nn.Module):
    def __init__(self, n_res_blocks=8):
        super(De_resnet, self).__init__()
        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.down_sample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU()
        )
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        block = self.block_input(x)
        for res_block in self.res_blocks:
            block = res_block(block)
        block = self.down_sample(block)
        block = self.block_output(block)
        return torch.sigmoid(block)


class De_resnet_bilinear(nn.Module):
    def __init__(self, n_res_blocks=8):
        super(De_resnet_bilinear, self).__init__()
        self.block_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(n_res_blocks)])
        self.down_sample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU()
        )
        self.block_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.bilinear = F.interpolate

    def forward(self, x):
        block = self.block_input(x)
        for res_block in self.res_blocks:
            block = res_block(block)
        block = self.bilinear(block, scale_factor=0.25, mode='bilinear')
        # block = self.down_sample(block)
        block = self.block_output(block)

        return torch.sigmoid(block)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT_Hc(nn.Module):
    def __init__(self, cs='sum', norm=True):
        super(DWT_Hc, self).__init__()
        self.requires_grad = False
        self.cs = cs
        self.norm = norm

    def forward(self, x):
        LL, Hc = dwt_init(x)
        if self.norm:
            Hc = Hc * 0.5 + 0.5
        if self.cs == 'sum':
            return LL, (Hc[:, :3, :, :] + Hc[:, 3:6, :, :] + Hc[:, 6:9, :, :])
        elif self.cs == 'mean':
            return LL, (Hc[:, :3, :, :] + Hc[:, 3:6, :, :] + Hc[:, 6:9, :, :]) / 3
        elif self.cs == 'cat':
            return LL, Hc


class DWT_LL(nn.Module):
    def __init__(self, cs='sum', norm=True):
        super(DWT_LL, self).__init__()
        self.requires_grad = False
        self.cs = cs
        self.norm = norm

    def forward(self, x):
        LL, Hc = dwt_init(x)
        if self.norm:
            LL = LL * 0.5
        return LL


class NLayerDiscriminator_stride1(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.InstanceNorm2d, padw=2, kw=4):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_stride1, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # kw = kw
        # padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]

        # sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=0), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class NLayerDiscriminator_stride2(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, padw=2, kw=5):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_stride2, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # kw = kw
        # padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=0), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



class Discriminator_wavelet(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False, wgan=False,
                 highpass=True, cs='cat', patchgan='s1'):
        super(Discriminator_wavelet, self).__init__()
        self.filter = DWT_Hc(cs)
        if cs == 'sum' or cs == 'mean':
            input_channel = 3
        elif cs == 'cat':
            input_channel = 9
        if patchgan == 's2':
            self.net = NLayerDiscriminator_stride2(input_nc=input_channel, ndf=64, n_layers=2,
                                                   norm_layer=nn.InstanceNorm2d, kw=4)
            print('NLayerDiscriminator s2 \n')
        elif patchgan == 's1':
            self.net = NLayerDiscriminator_stride1(input_nc=input_channel, ndf=64, n_layers=2,
                                                   norm_layer=nn.InstanceNorm2d, kw=4)
            print('NLayerDiscriminator s1 \n')
        else:
            self.net = DiscriminatorBasic(n_input_channels=input_channel)
            print('DiscriminatorBasic \n')
        self.wgan = wgan

    def forward(self, x, y=None):
        if self.filter is not None:
            _, x = self.filter(x)
        x = self.net(x)
        if y is not None:
            _, y = self.filter(x)
            x -= self.net(y).mean(0, keepdim=True)
        if not self.wgan:
            x = torch.sigmoid(x)
        return x

    def filter_wavelet(self, x, norm=True):
        DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        LL, Hc = DWT2(x)
        LH, HL, HH = Hc[0][0, :, 0, :, :], Hc[0][0, :, 1, :, :], Hc[0][0, :, 2, :, :]

        if norm:
            LL = LL * 0.5
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs == 'sum':
            return LL, (LH+HL+HH)/3.
        elif self.cs == 'cat':
            return LL, torch.cat((LH, HL, HH), 1)





class Discriminator_Gau(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False, wgan=False, highpass=True,
                 patchgan=False):
        super(Discriminator_Gau, self).__init__()
        if highpass:
            self.filter = FilterHigh(recursions=recursions, stride=stride, kernel_size=kernel_size, include_pad=False,
                                     gaussian=gaussian)
        else:
            self.filter = None
        if patchgan:
            self.net = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=2, norm_layer=nn.InstanceNorm2d)
            print('NLayerDiscriminator \n')
        else:
            self.net = DiscriminatorBasic(n_input_channels=3)
            print('DiscriminatorBasic \n')
        self.wgan = wgan

    def forward(self, x, y=None):
        if self.filter is not None:
            x = self.filter(x)
        x = self.net(x)
        if y is not None:
            x -= self.net(self.filter(y)).mean(0, keepdim=True)
        if not self.wgan:
            x = torch.sigmoid(x)
        return x


class DiscriminatorBasic(nn.Module):
    def __init__(self, n_input_channels=3):
        super(DiscriminatorBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img





if __name__ == '__main__':
    # D = Discriminator_wavelet(cs='cat', patchgan=True)
    D = Discriminator_wavelet(cs='cat', patchgan='s2')
    checkpoint = torch.load('/media/4T/Dizzy/real-world-sr/dsgan/checkpoints/checkpoints/0617_DeResnet+wavecat+Nlayer+wtex0.1/iteration_132801.tar')
    D.load_state_dict(checkpoint['models_d_state_dict'])
