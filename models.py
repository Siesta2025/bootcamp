import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class BasicBlock(nn.Module):
    # (B, C_in, H, W) -> (B, C_out, H/stride, W/stride)
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.needs_projection = (in_channels != out_channels) or (stride != 1)
        if self.needs_projection:
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.bn_proj = nn.BatchNorm2d(out_channels)
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
        if self.needs_projection:
            identity = self.projection(x)
            identity = self.bn_proj(identity)
        else:
            identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SmallResNet(nn.Module):
    # essentially an encoder, (B, C_in, H, W) -> (B, C_out)
    # specifically (B, 256)
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        self.stage2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )

        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim=256, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        h = self.encoder(x)
        logits =  self.head(h)
        return logits

class ProjectionHead(nn.Module):
    # (B, C_in) -> (B, C_out)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x 
    
class SimCLRModel(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x




    


    (view1_batch, view2_batch), labels = next(iter(loader))

    print("view1_batch:", view1_batch.shape) # (8, 3, 32, 32)
    print("view2_batch:", view2_batch.shape) # the same
    print("labels", labels.shape) # (8)

    encoder = SmallResNet()
    projector = ProjectionHead(256, 128)
    model = SimCLRModel(encoder, projector)

    z1 = model(view1_batch)
    z2 = model(view2_batch)
    print("z1 & z2 shape:", z1.shape)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    print("z shape:", z.shape)
    print("normalization result:", z.norm(dim=1))

    similarity = z @ z.T
    print("similarity matrix shape:", similarity.shape)
    print("diagonals:", similarity.diag())


