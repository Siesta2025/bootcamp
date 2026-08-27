import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        needs_projection = (in_channels != out_channels) or (stride != 1)
        if needs_projection:
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.bn_proj = nn.BatchNorm2d(out_channels)
        else:
            self.projection = nn.Identity()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.projection(x)
        identity = self.bn_proj(identity)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


if __name__ == "__main__":
    x = torch.randn(8, 64, 32, 32)  

    block1 = BasicBlock(64, 64, 1)
    y1 = block1(x)

    block2 = BasicBlock(64, 128, 2)
    y2 = block2(x)

    print(y1.shape)
    print(y2.shape)