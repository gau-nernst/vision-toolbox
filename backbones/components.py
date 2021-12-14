from torch import nn

class ConvBnAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_fn()
        
        mode = "fan_out"
        if isinstance(self.act, nn.ReLU):
            nn.init.kaiming_normal_(self.conv.weight, mode=mode, nonlinearity="relu")
        elif isinstance(self.act, nn.LeakyReLU):
            nn.init.kaiming_normal_(self.conv.weight, a=self.act.negative_slope, mode=mode, nonlinearity="leaky_relu")
