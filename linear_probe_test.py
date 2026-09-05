import torch
import torch.nn as nn
import math
from torch.utils.data import Subset
from linear_probe import LinearProbe
from models import SmallResNet
from linear_probe_train import load_encoder_checkpoint, get_train_val_dataset, get_test_dataset, load_data, train_one_epoch, evaluate

if __name__ == "__main__":
    # test 1: LinearProbe module, encoder checkpoint loading, encoder freezing
    encoder = SmallResNet()
    encoder = load_encoder_checkpoint(encoder, "./outputs/simclr_30/best.pt")

    prober = LinearProbe(encoder, num_classes=10)
    for param in prober.frozen_encoder.parameters():
        assert param.requires_grad is False
    for param in prober.classifier.parameters():
        assert param.requires_grad is True
    prober.frozen_encoder.eval()
    prober.classifier.train()

    # test 2: forward pass, loss backward, optimizer
    x = torch.randn(8, 3, 32, 32)
    label = torch.randint(0, 10, (8, ))
    output = prober(x)
    assert output.shape == (8, 10)

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

    # test3: data loading
    train_val_dataset = get_train_val_dataset("./data")
    test_dataset = get_test_dataset("./data")
    indices = torch.randperm(50000)
    train_indices = indices[:45000]
    val_indices = indices[45000:]
    train_dataset = Subset(
        train_val_dataset,
        train_indices,
    )
    val_dataset = Subset(
        train_val_dataset,
        val_indices,
    )

    train_loader = load_data(
        train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=0,
    )
    val_loader = load_data(
        val_dataset,
        shuffle=False,
        batch_size=8,
        num_workers=0,
    )
    test_loader = load_data(
        test_dataset,
        shuffle=False,
        batch_size=8,
        num_workers=0,
    )
    train_x, train_y = next(iter(train_loader))
    assert train_x.shape == (8, 3, 32, 32) and train_y.shape == (8, )
    val_x, val_y = next(iter(val_loader))
    assert val_x.shape == (8, 3, 32, 32) and val_y.shape == (8, )
    test_x, test_y = next(iter(test_loader))
    assert test_x.shape == (8, 3, 32, 32) and test_y.shape == (8, )

    # test 4: train_one_epoch
    encoder = SmallResNet()
    prober = LinearProbe(encoder, num_classes=10)

    temp_train_loader = load_data(
        Subset(
            train_dataset,
            list(range(16))
        ),
        shuffle=True,
        batch_size=8,
        num_workers=0,
    )

    optimizer = torch.optim.SGD(
        prober.classifier.parameters(),
        lr=0.1,
        momentum=0.9,
    )

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loss, train_accuracy = train_one_epoch(
        prober,
        temp_train_loader,
        optimizer,
        criterion,
        device
    )
    print("loss1:", train_loss, "acc1:", train_accuracy)
    assert isinstance(train_loss, float) and math.isfinite(train_loss) and 0<=train_accuracy<=1.0

    train_loss, train_accuracy = train_one_epoch(
        prober,
        temp_train_loader,
        optimizer,
        criterion,
        device
    )
    print("loss2:", train_loss, "acc2:", train_accuracy)
    assert isinstance(train_loss, float) and math.isfinite(train_loss) and 0<=train_accuracy<=1.0

    # test 5: evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SmallResNet()
    prober = LinearProbe(encoder, num_classes=10).to(device)

    temp_val_loader = load_data(
        Subset(
            val_dataset,
            list(range(16))
        ),
        shuffle=True,
        batch_size=8,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    before = next(prober.classifier.parameters()).detach().clone()
    train_loss, train_accuracy = evaluate(
        prober,
        temp_val_loader,
        criterion,
        device
    )
    after = next(prober.classifier.parameters()).detach().clone()
    assert torch.allclose(before, after)
    print("loss:", train_loss, "acc:", train_accuracy)
