import torch
from torch import nn


class IGCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = nn.Sequential(
            # gcn1
            nn.Conv2d(5, 16, 1, 1, 0, bias=True),  # -1 x 5 x K x K -> -1 x 16 x K x K
            nn.ReLU(),
            # gcn2
            nn.Conv2d(16, 16, 1, 1, 0, bias=True),  # -1 x 16 x K x K -> -1 x 16 x K x K
            nn.ReLU(),
            # gcn3
            nn.Conv2d(16, 6, 1, 1, 0, bias=True),  # -1 x 16 x K x K -> -1 x 6 x K x K
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(15, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 1, bias=True)
        )

    def forward(self, x, Xdiag, intens, w_alpha):
        # MLP1
        x = self.gcn(x)
        fea_mean = torch.sum(x, dim=3)
        fea_max, _ = torch.max(x, dim=3)
        fea = torch.cat((fea_mean, fea_max), dim=1)  # -1 x 6 x K x K
        # 整合输入到 MLP2
        mlp2In = torch.cat((Xdiag, intens, fea, w_alpha), dim=1).transpose(1, 2).contiguous()  # -1 x 15 x K -> -1 x K x 15
        # MLP2
        out = self.mlp2(mlp2In) # -1 x K x 1
        pred = torch.sigmoid(out)  # -1 x K x 1
        # pred = pred.transpose(1, 2).contiguous()  # -1 x 1 x K
        return pred


def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.normal_(layer.bias, std=0.1)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


if __name__ == "__main__":
    K = 20
    igcn = IGCNet()
    igcn.apply(init_weights)
    x = torch.randn(size=(64, 5, K, K))
    Xdiag = torch.rand(size=(64, 1, K))
    intens = torch.rand(size=(64, 1, K))
    w_alpha = torch.rand(size=(64, 1, K))

    output = igcn(x, Xdiag, intens, w_alpha)
    print(output.shape)

