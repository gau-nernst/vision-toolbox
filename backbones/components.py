import torch
from torch import nn
import torch.nn.functional as F

class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, act_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_fn(inplace=True)
        
        mode = "fan_out"
        if isinstance(self.act, (nn.ReLU, nn.ReLU6)):
            nn.init.kaiming_normal_(self.conv.weight, mode=mode, nonlinearity="relu")
        elif isinstance(self.act, nn.LeakyReLU):
            nn.init.kaiming_normal_(self.conv.weight, a=self.act.negative_slope, mode=mode, nonlinearity="leaky_relu")

class SeperableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, act_fn=nn.ReLU6):
        super().__init__()
        self.dw = ConvBnAct(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, act_fn=act_fn)
        self.pw = ConvBnAct(in_channels, out_channels, kernel_size=1, padding=0, act_fn=act_fn)

class ESE(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.linear = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1,1))
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        
        return x * out

class SPP(nn.Module):
    def __init__(self, num_kernels=None, pool_fn="max"):
        assert pool_fn in ("max", "avg")
        super().__init__()
        if num_kernels is None:
            num_kernels = [5, 9, 13]

        pool_fn = nn.MaxPool2d if pool_fn == "max" else nn.AvgPool2d
        pools = [pool_fn(k, stride=1, padding=int(round((k-1)/2))) for k in num_kernels]
        self.pools = nn.ModuleList(pools)
    
    def forward(self, x):
        outputs = [pool(x) for pool in self.pools]
        out = torch.concat(outputs)

        return out
