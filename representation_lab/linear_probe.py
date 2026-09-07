import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()

        self.frozen_encoder = encoder
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(256, num_classes)  

    def forward(self, x):
        output = self.frozen_encoder(x)
        output = self.classifier(output)
        return output

