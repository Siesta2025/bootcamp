import torch
import math

from torchvision import transforms
from torch.utils.data import Subset
from representation_lab.classifier_data import get_train_val_dataset, get_test_dataset, load_data
from representation_lab.models import Classifier, SmallResNet
from representation_lab.training import train_supervised_one_epoch, evaluate_classifier

# test1: dataset and data loading
train_dataset = get_train_val_dataset("./data", transforms.ToTensor())
val_dataset = get_train_val_dataset("./data", transforms.ToTensor())
test_dataset = get_test_dataset("./data", transforms.ToTensor())
train_sample = next(iter(train_dataset))
val_sample = next(iter(val_dataset))
test_sample = next(iter(test_dataset))
assert train_sample[0].shape == (3, 32, 32)
assert val_sample[0].shape == (3, 32, 32)
assert test_sample[0].shape == (3, 32, 32)

train_loader = load_data(
    Subset(train_dataset, list(range(8))),
    shuffle=True,
    batch_size=8,
    num_workers=0,
)
x1, y1 = next(iter(train_loader))
assert x1.shape == (8, 3, 32, 32) and y1.shape == (8,)
test_loader = load_data(
    Subset(test_dataset, list(range(8))),
    shuffle=False,
    batch_size=8,
    num_workers=0,
)
x2, y2 = next(iter(test_loader))
assert x2.shape == (8, 3, 32, 32) and y2.shape == (8,)

# test2: model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = SmallResNet()
model = Classifier(encoder).to(device)

x = torch.randn(8, 3, 32, 32).to(device)
logits = model(x)
assert logits.shape == (8, 10)

# test3: training loop
train_dataset = get_train_val_dataset("./data", transforms.ToTensor())
train_loader = load_data(
    Subset(train_dataset, list(range(8))),
    shuffle=True,
    batch_size=8,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = SmallResNet()
model = Classifier(encoder).to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)

before = next(model.parameters()).detach().clone()
loss, acc = train_supervised_one_epoch(model, train_loader, criterion, optimizer, device)
after = next(model.parameters()).detach().clone()

assert isinstance(loss, float) and math.isfinite(loss)
assert 0.0 <= acc <= 1.0
assert not torch.allclose(before, after)

# test4: evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = get_train_val_dataset("./data", transforms.ToTensor())
val_loader = load_data(
    Subset(val_dataset, list(range(8))),
    shuffle=False,
    batch_size=8,
    num_workers=0,
)

criterion = torch.nn.CrossEntropyLoss()

loss, acc = evaluate_classifier(model, val_loader, criterion, device)
assert isinstance(loss, float) and math.isfinite(loss)
