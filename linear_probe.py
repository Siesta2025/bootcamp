import torch
import torch.nn as nn
from models import SmallResNet
from reload_encoder_checkpoint import reload_encoder_checkpoint

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

if __name__ == "__main__":
    encoder = SmallResNet()
    encoder = reload_encoder_checkpoint(encoder, "./outputs/simclr_30/best.pt")

    prober = LinearProbe(encoder, num_classes=10)
    for param in prober.frozen_encoder.parameters():
        assert param.requires_grad is False
    for param in prober.classifier.parameters():
        assert param.requires_grad is True
    prober.frozen_encoder.eval()
    prober.classifier.train()

    x = torch.randn(8, 3, 32, 32)
    label = torch.randint(0, 10, (8, ))
    output = prober(x)

    optimizer = torch.optim.SGD(
        prober.classifier.parameters(),
        lr = 0.1,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    loss = criterion(output, label)
    loss.backward()
    for param in prober.frozen_encoder.parameters():
        assert param.grad is None
    for param in prober.classifier.parameters():
        assert param.grad is not None

    encoder_param_sample = next(prober.frozen_encoder.parameters())
    classifier_param_sample = next(prober.classifier.parameters())
    encoder_param_sample_before = encoder_param_sample.detach().clone()
    classifier_param_sample_before = classifier_param_sample.detach().clone()
    optimizer.step()
    encoder_param_sample_after = encoder_param_sample.detach().clone()
    classifier_param_sample_after = classifier_param_sample.detach().clone()
    assert torch.allclose(encoder_param_sample_before, encoder_param_sample_after)
    assert not torch.allclose(classifier_param_sample_before, classifier_param_sample_after)




    