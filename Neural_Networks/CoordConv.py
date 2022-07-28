import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import numpy as np

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
# Try using gpu instead of cpu
device = try_gpu()

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor, device):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """

        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            xx_channel = xx_channel.to(device)
            yy_channel = yy_channel.to(device)
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            xx_channel = xx_channel.to(device)
            yy_channel = yy_channel.to(device)
            zz_channel = zz_channel.to(device)
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, device=device):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor, device):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor, device)
        out = self.conv(out)

        return out

class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, device=device):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor, device)
        out = self.conv(out)

        return out
        
class Net_CoordConv(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_features=128, device=device):
        super(Net_CoordConv, self).__init__()
        self.coordconv = CoordConv3d(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        self.conv1 = nn.Conv3d(hidden_dim[0],  hidden_dim[1], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim[1], hidden_dim[2], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(hidden_dim[2], hidden_dim[3], kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(hidden_dim[3], hidden_dim[4], kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(5*5*3*hidden_dim[4], out_features)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.coordconv(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Res3dBlock_d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
              
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()           

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU() 
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.conv1(input))
        input = self.relu1(input)
        input = self.bn2(self.conv2(input))
        input = input + shortcut
        return self.relu2(input)
        
        
class ResNet18_noAnoP_CoordConv_torch(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_d, out_features=128, device=device):
        super().__init__()
        
        self.coordconv = CoordConv3d(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        
        self.layer0 = nn.Sequential(
            nn.Conv3d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim[0]),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[0], downsample=False),
            resblock(hidden_dim[0], hidden_dim[0], downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[1], downsample=True),
            resblock(hidden_dim[1], hidden_dim[1], downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(hidden_dim[1], hidden_dim[2], downsample=True),
            resblock(hidden_dim[2], hidden_dim[2], downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(hidden_dim[2], hidden_dim[3], downsample=True),
            resblock(hidden_dim[3], hidden_dim[3], downsample=False)
        )
  
        self.fc2 = torch.nn.Linear(5*5*3*hidden_dim[3], out_features)

    def forward(self, input):
        input = self.coordconv(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = input.view(input.size(0), -1)
        input = self.fc2(input)

        return input   

"""
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

class AddCoords_keras(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords_keras, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor, device):
        '''
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        '''

        if self.rank == 2:
            batch_shape, channels, dim1, dim2, dim3  = input_tensor.shape
            
            xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            xx_channels = torch.Tensor(xx_channels.numpy())
            yy_channels = torch.Tensor(yy_channels.numpy())
          
            xx_channels = xx_channels.to(device)
            yy_channels = yy_channels.to(device)
            
            out = torch.cat([input_tensor, xx_channels, yy_channels], dim=1)

        elif self.rank == 3:
            batch_shape, channels, dim1, dim2, dim3 = input_tensor.shape
            
            xx_ones = K.ones(K.stack([batch_shape, dim3]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)

            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            xx_channels = K.expand_dims(xx_channels, axis=1)
            xx_channels = K.tile(xx_channels,
                                 [1, dim1, 1, 1, 1])

            yy_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim3), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            yy_channels = K.expand_dims(yy_channels, axis=1)
            yy_channels = K.tile(yy_channels,
                                 [1, dim1, 1, 1, 1])

            zz_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            zz_range = K.expand_dims(zz_range, axis=-1)
            zz_range = K.expand_dims(zz_range, axis=-1)

            zz_channels = K.tile(zz_range,
                                 [1, 1, dim2, dim3])
            zz_channels = K.expand_dims(zz_channels, axis=-1)

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim2 - 1, K.floatx())
            xx_channels = xx_channels * 2 - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim3 - 1, K.floatx())
            yy_channels = yy_channels * 2 - 1.

            zz_channels = K.cast(zz_channels, K.floatx())
            zz_channels = zz_channels / K.cast(dim1 - 1, K.floatx())
            zz_channels = zz_channels * 2 - 1.

            xx_channels = np.swapaxes(xx_channels.numpy(),3,4)
            xx_channels = np.swapaxes(xx_channels,2,3)
            xx_channels = torch.Tensor(np.swapaxes(xx_channels,1,2))

            yy_channels = np.swapaxes(yy_channels.numpy(),3,4)
            yy_channels = np.swapaxes(yy_channels,2,3)
            yy_channels = torch.Tensor(np.swapaxes(yy_channels,1,2))

            zz_channels = np.swapaxes(zz_channels.numpy(),3,4)
            zz_channels = np.swapaxes(zz_channels,2,3)
            zz_channels = torch.Tensor(np.swapaxes(zz_channels,1,2))
            
            xx_channels = xx_channels.to(device)
            yy_channels = yy_channels.to(device)
            zz_channels = zz_channels.to(device)
                
            out = torch.cat([input_tensor, xx_channels, yy_channels, zz_channels], dim=1)

        else:
            raise NotImplementedError

        return out
        

class CoordConv3d_keras(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, device=device):
        super(CoordConv3d_keras, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords_keras(self.rank, with_r)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        '''
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        '''
        out = self.addcoords(input_tensor, device)
        out = self.conv(out)

        return out
        

        
class Net_CoordConv_keras(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_features=128, device=device):
        super(Net_CoordConv_keras, self).__init__()
        self.coordconv = CoordConv3d_keras(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        self.conv1 = nn.Conv3d(hidden_dim[0],  hidden_dim[1], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim[1], hidden_dim[2], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(hidden_dim[2], hidden_dim[3], kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(hidden_dim[3], hidden_dim[4], kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(5*5*3*hidden_dim[4], out_features)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.coordconv(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
      
                   
class ResNet18_d_CoordConv_keras(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_d, out_features=128):
        super().__init__()
        
        self.coordconv = CoordConv3d_keras(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        
        self.layer0 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim[0]),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[0], downsample=False),
            resblock(hidden_dim[0], hidden_dim[0], downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[1], downsample=True),
            resblock(hidden_dim[1], hidden_dim[1], downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(hidden_dim[1], hidden_dim[2], downsample=True),
            resblock(hidden_dim[2], hidden_dim[2], downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(hidden_dim[2], hidden_dim[3], downsample=True),
            resblock(hidden_dim[3], hidden_dim[3], downsample=False)
        )
  
        self.gap = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = torch.nn.Linear(hidden_dim[3], out_features)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)

        return input


class ResNet18_noAnoP_CoordConv_keras(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_d, out_features=128, device=device):
        super().__init__()
        
        self.coordconv = CoordConv3d_keras(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        
        self.layer0 = nn.Sequential(
            nn.Conv3d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim[0]),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[0], downsample=False),
            resblock(hidden_dim[0], hidden_dim[0], downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[1], downsample=True),
            resblock(hidden_dim[1], hidden_dim[1], downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(hidden_dim[1], hidden_dim[2], downsample=True),
            resblock(hidden_dim[2], hidden_dim[2], downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(hidden_dim[2], hidden_dim[3], downsample=True),
            resblock(hidden_dim[3], hidden_dim[3], downsample=False)
        )
  
        self.fc2 = torch.nn.Linear(5*5*3*hidden_dim[3], out_features)

    def forward(self, input):
        input = self.coordconv(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = input.view(input.size(0), -1)
        input = self.fc2(input)

        return input       
        


class Res3dBlock_nobatchnorm(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
              
        if downsample:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2),
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()           
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU() 
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.conv1(input)
        input = self.relu1(input)
        input = self.conv2(input)
        input = input + shortcut
        return self.relu2(input)

class ResNet18_CoordConv_no_batchnorm(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_nobatchnorm, out_features=128):
        super().__init__()
        
        self.coordconv = CoordConv3d_keras(in_channels,  hidden_dim[0], kernel_size=7, stride=2, padding=3, with_r=False)
        
        self.layer0 = nn.Sequential(
            nn.Conv3d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[0], downsample=False),
            resblock(hidden_dim[0], hidden_dim[0], downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(hidden_dim[0], hidden_dim[1], downsample=True),
            resblock(hidden_dim[1], hidden_dim[1], downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(hidden_dim[1], hidden_dim[2], downsample=True),
            resblock(hidden_dim[2], hidden_dim[2], downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(hidden_dim[2], hidden_dim[3], downsample=True),
            resblock(hidden_dim[3], hidden_dim[3], downsample=False)
        )
  
        self.fc = torch.nn.Linear(5*5*3*hidden_dim[3], out_features)
       
    def forward(self, input):
        input = self.coordconv(input)
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)

        return input
"""