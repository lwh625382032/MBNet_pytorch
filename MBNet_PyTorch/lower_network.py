import torch
import torch.nn as nn
#
# # class DnCNN(nn.Module):
# #     def __init__(self, channels, num_of_layers=17):
# #         super(DnCNN, self).__init__()
# #         kernel_size = 3
# #         padding = 1
# #         features = 64
# #         layers = []
# #         layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
# #         layers.append(nn.ReLU(inplace=True))
# #         for _ in range(num_of_layers-2):
# #             layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
# #             layers.append(nn.BatchNorm2d(features))
# #             layers.append(nn.ReLU(inplace=True))
# #         layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
# #         self.dncnn = nn.Sequential(*layers)
# #     def forward(self, x):
# #         out = self.dncnn(x)
# #         return out


class DnCNN(nn.Module):
    def __init__(self, channels):

        super(DnCNN, self).__init__()

        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=5, dilation=5,bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=4, dilation=4,bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        n = self.model(x)
        return n


def get_layer_param(model):
    return sum([torch.numel(param) for param in model.parameters()])


if __name__ == '__main__':
    model = DnCNN(1)
    x = torch.ones(1,1,50,50)
    out = model(x)

    y = get_layer_param(model)
    print(model)
    print(out.shape)
    print("模型的参数量为：%d" % y)
    # writer = SummaryWriter(comment='DnCNN')
    # writer.add_graph(model,x)