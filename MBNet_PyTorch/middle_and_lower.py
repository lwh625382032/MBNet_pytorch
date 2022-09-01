from logging import handlers

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
class BRDNet(nn.Module):
    def __init__(self, channels):
        super(BRDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1

        """
        第一条分支
        """
        # layers = []
        #
        # # 第1层
        # layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        # layers.append(nn.BatchNorm2d(features))
        # layers.append(nn.ReLU(inplace=True))
        # # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        #
        # # 第2-16层
        # for _ in range(15):
        #     layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        #     layers.append(nn.BatchNorm2d(features))
        #     layers.append(nn.ReLU(inplace=True))
        # # layers.append(B.ResBlock(features, features, kernel_size=kernel_size, stride=1, padding=padding, bias=False, mode='CRC', negative_slope=0.2))
        #
        # # 第17层
        # layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,bias=False))  # 原来out_channels为channels

        """
        第二条分支
        """
        L = []

        # 第1层
        L.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))

        # 第2-8层
        for i in range(7):
            L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,bias=False, dilation=2))
            L.append(nn.BatchNorm2d(features))
            L.append(nn.ReLU(inplace=True))

        # 第9层
        L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))

        # 第10-15层
        for i in range(6):
            L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2,groups=groups,bias=False, dilation=2))
            L.append(nn.BatchNorm2d(features))
            L.append(nn.ReLU(inplace=True))

        # 第16层
        L.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,bias=False))
        L.append(nn.BatchNorm2d(features))
        L.append(nn.ReLU(inplace=True))

        # 第17层
        L.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,bias=False))

        """
        第三条分支
        """

        layer1 = []

        # 第1层
        layer1.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, stride=1, padding=1, dilation=1,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第2层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=2, dilation=2,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第3层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=3, dilation=3,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第4层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=4, dilation=4,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第5层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=5, dilation=5,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第6层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=4, dilation=4,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第7层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=3, dilation=3,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第8层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=2, dilation=2,bias=True))
        layer1.append(nn.BatchNorm2d(features))
        layer1.append(nn.ReLU(inplace=True))

        # 第9层
        layer1.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1,bias=True))

        ################################################################################################################################

        # self.BRDNet_first = nn.Sequential(*layers)
        self.BRDNet_second = nn.Sequential(*L)
        self.BRDNet_third = nn.Sequential(*layer1)
        # 第18层
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=kernel_size,
                                             padding=padding, groups=groups, bias=False))
    def forward(self, x):
        # out1 = self.BRDNet_first(x)
        out2 = self.BRDNet_second(x)
        out3 = self.BRDNet_third(x)
        # out1 = x - out1
        out2 = x - out2
        out3 = x - out3
        out = torch.cat([out2,out3], 1)
        out = self.conv1(out)    ## out 为 残差图像，近似为 噪声
        # out = x - out         ## out 为 得到的干净图像 ，近似为 原图
        return out


def get_layer_param(model):
    return sum([torch.numel(param) for param in model.parameters()])


if __name__ == '__main__':
    model = BRDNet(1)
    x = torch.ones(1,1,50,50)
    out = model(x)

    y = get_layer_param(model)
    print(model)
    print(out.shape)
    print("模型的参数量为：%d" % y)
    # writer = SummaryWriter(comment='DnCNN')
    # writer.add_graph(model,x)