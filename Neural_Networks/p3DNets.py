import torch
from torch import nn

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

class Res3dBlock_lrelu(nn.Module):
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

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1) 
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.conv1(input))
        input = self.relu1(input)
        input = self.bn2(self.conv2(input))
        input = input + shortcut
        return self.relu2(input)
        
        
class Res3dBlock_lrelu_final(nn.Module):
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

        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1) 
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.conv1(input)
        input = self.relu1(input)
        input = self.conv2(input)
        input = input + shortcut
        return input

                    
class ResNet18_3D_d(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_features, resblock=Res3dBlock_d):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim[0], kernel_size=7, stride=2, padding=3),
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
           
class ResNet18_3D_noadaptive_nopool(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_d, out_features=128):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim[0], kernel_size=7, stride=2, padding=3),
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
  
        #self.fc1 = torch.nn.Linear(5*5*3*hidden_dim[3], hidden_dim[4])
        self.fc2 = torch.nn.Linear(5*5*3*hidden_dim[3], out_features)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = input.view(input.size(0), -1)
        #input = self.fc1(input)
        input = self.fc2(input)

        return input       


class ResNet18_3D_noadaptive_nopool_lrelu(nn.Module):
    def __init__(self, in_channels, hidden_dim, resblock=Res3dBlock_lrelu, resblock2=Res3dBlock_lrelu_final, out_features=128):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim[0], kernel_size=7, stride=2, padding=3),
            nn.Conv3d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim[0]),
            nn.LeakyReLU(0.1)
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
            resblock2(hidden_dim[3], hidden_dim[3], downsample=False)
        )
  
        #self.fc1 = torch.nn.Linear(5*5*3*hidden_dim[3], hidden_dim[4])
        self.fc2 = torch.nn.Linear(5*5*3*hidden_dim[3], out_features)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = input.view(input.size(0), -1)
        #input = self.fc1(input)
        input = self.fc2(input)

        return input       
        
class VGG3dBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool, activation):
        super().__init__()  
            
        if activation:
            self.act1 =  nn.ReLU()
            self.act2 =  nn.ReLU()
        else:
            self.act1 =  nn.Tanh()
            self.act2 =  nn.Tanh()             
            
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if maxpool:
            self.max_pool = nn.MaxPool3d(2)
        
        else: 
            self.max_pool = nn.AvgPool3d(2)

    def forward(self, input):
        input = self.conv1(input)
        input = self.act1(input)
        input = self.conv2(input)
        input = self.act2(input)
               
        return self.max_pool(input)        
        
class VGG3dBlock_3(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool, activation):
        super().__init__()  
            
        if activation:
            self.act1 =  nn.ReLU()
            self.act2 =  nn.ReLU()
            self.act3 =  nn.ReLU()
        else:
            self.act1 =  nn.Tanh()
            self.act2 =  nn.Tanh()    
            self.act3 =  nn.Tanh()            
            
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if maxpool:
            self.max_pool = nn.MaxPool3d(2)
        
        else: 
            self.max_pool = nn.AvgPool3d(2)

    def forward(self, input):
        input = self.conv1(input)
        input = self.act1(input)
        input = self.conv2(input)
        input = self.act2(input)
        input = self.conv3(input)
        input = self.act3(input)
               
        return self.max_pool(input)
        
class VGG16_3D(nn.Module):
    def __init__(self, in_channels, hidden_channels, maxpool, activation, vgg_2=VGG3dBlock_2, vgg_3=VGG3dBlock_3, out_features=128):
        super().__init__()
        
        self.layer1 = vgg_2(in_channels, hidden_channels[0], maxpool, activation)

        self.layer2 = vgg_2(hidden_channels[0], hidden_channels[1], maxpool, activation)
        
        self.layer3 = vgg_3(hidden_channels[1], hidden_channels[2], maxpool, activation)
        
        self.layer4 = vgg_3(hidden_channels[2], hidden_channels[3], maxpool, activation)
  
        self.layer5 = vgg_3(hidden_channels[3], hidden_channels[4], maxpool, activation)
        
        self.conv6 = nn.Conv3d(hidden_channels[4], hidden_channels[5], 
                                     kernel_size=(3,1,1), stride=1, padding=0)
        self.relu6 = nn.ReLU()
        
        if activation: 
            self.linear_layers = nn.Sequential(
                nn.Linear(5*5*hidden_channels[5], hidden_channels[6]),
                nn.ReLU(),
                nn.Linear(hidden_channels[6], hidden_channels[7]),
                nn.ReLU(),
                nn.Linear(hidden_channels[7], out_features)
            )
        else:
            self.linear_layers = nn.Sequential(
                nn.Linear(5*5*hidden_channels[5], hidden_channels[6]),
                nn.Tanh(),
                nn.Linear(hidden_channels[6], hidden_channels[7]),
                nn.Tanh(),
                nn.Linear(hidden_channels[7], out_features)
            )
 
    def forward(self, input):
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.conv6(input)
        input = self.relu6(input)
        input = input.view(input.size(0), -1)
        input = self.linear_layers(input)

        return input
    
class VGG16_3D_endave(nn.Module):
    def __init__(self, in_channels, hidden_channels, maxpool, activation, vgg_2=VGG3dBlock_2, vgg_3=VGG3dBlock_3, out_features=128):
        super().__init__()
        
        self.layer1 = vgg_2(in_channels, hidden_channels[0], maxpool, activation)

        self.layer2 = vgg_2(hidden_channels[0], hidden_channels[1], maxpool, activation)
        
        self.layer3 = vgg_3(hidden_channels[1], hidden_channels[2], maxpool, activation)
        
        self.layer4 = vgg_3(hidden_channels[2], hidden_channels[3], maxpool, activation)
  
        self.layer5 = vgg_3(hidden_channels[3], hidden_channels[4], maxpool, activation)
        
        self.conv6 = nn.Conv3d(hidden_channels[4], hidden_channels[5], 
                                     kernel_size=(3,1,1), stride=1, padding=0)
        self.relu6 = nn.ReLU()
        
        if activation: 
            self.linear_layers = nn.Sequential(
                nn.Linear(hidden_channels[5], hidden_channels[6]),
                nn.ReLU(),
                nn.Linear(hidden_channels[6], hidden_channels[7]),
                nn.ReLU(),
                nn.Linear(hidden_channels[7], out_features)
            )
        else:
            self.linear_layers = nn.Sequential(
                nn.Linear(hidden_channels[5], hidden_channels[6]),
                nn.Tanh(),
                nn.Linear(hidden_channels[6], hidden_channels[7]),
                nn.Tanh(),
                nn.Linear(hidden_channels[7], out_features)
            )
            
        self.max_pool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, input):
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.conv6(input)
        input = self.relu6(input)
        input = self.max_pool(input)
        input = input.view(input.size(0), -1)
        input = self.linear_layers(input)
            
        return input