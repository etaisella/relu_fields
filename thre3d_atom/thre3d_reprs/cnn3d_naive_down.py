""" Parts of the U-Net model """
import torch.nn as nn

class cnn3d_naive_down(nn.Module):
  def __init__(self, n_channels, small=False):
    super(cnn3d_naive_down, self).__init__()
    if small:
        df = 4
    else:
        df = 2
    self.conv1 = nn.Conv3d(n_channels, 64, (3, 3, 3), stride=(1,1,1), padding=1)
    self.conv2 = nn.Conv3d(64, 64, (3, 3, 3), stride=(1,1,1), padding=1)
    self.conv3 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1,1,1), padding=1)
    self.conv4 = nn.Conv3d(128, n_channels, (3, 3, 3), stride=(1,1,1), padding=1)
    self.avgpool = nn.AvgPool3d((df,df,df), stride=(df,df,df))
    self.leakyRelu = nn.LeakyReLU(0.2)
  
  def forward(self, x):
    y = self.avgpool(x)
    x = self.conv1(x)
    x = self.leakyRelu(x)
    x = self.conv2(x)
    x = self.avgpool(x)
    x = self.conv3(x)
    x = self.leakyRelu(x)
    x = self.conv4(x)
    x = self.leakyRelu(x)
    return y + 0.0000001 * x