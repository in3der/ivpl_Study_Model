import torch
import torch.nn as nn
from torchsummary import summary

def make_conv(in_channels, out_channels, kernel_size, stride, padding, use_maxpool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
        ]
    if use_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*layers)

# 24 Conv layers + 2 FC layers
class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, S=7, B=2, C=20, **kwargs):
        super(YOLOv1, self).__init__()
        self.conv = nn.Sequential(
            make_conv(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, use_maxpool=True),
            
            make_conv(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, use_maxpool=True),
            
            make_conv(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, use_maxpool=True),

            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, use_maxpool=True),
            
            make_conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            make_conv(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            make_conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(S*S*1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, S*S*((1+4)*B+C)),
        )
        
        self.init_weights()
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        out = self.fc2(x)
        return out
    
    def init_weights(self):
        # Initialize all Conv2D and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

  
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = YOLOv1().to(device)
#summary(model, input_size=(3, 448, 448))