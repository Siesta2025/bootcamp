import torch

from representation_lab.models import SmallResNet, ProjectionHead, SimCLRModel
from representation_lab.simclr_data import get_train_val_dataset, load_data
from representation_lab.losses import NTXent

# test1
x = torch.randn(8, 3, 32, 32)
encoder = SmallResNet()
h = encoder(x)
assert h.shape == (8, 256)
projection_head = ProjectionHead(256, 128)
z = projection_head(h)
assert z.shape == (8, 128)

# test2
z1 = torch.randn(8, 128, requires_grad=True)
z2 = torch.randn(8, 128, requires_grad=True)

criterion = NTXent(tau=0.7)
loss = criterion(z1, z2)
assert loss.shape == torch.Size([]) # scalar value
assert torch.isfinite(loss)
assert loss.requires_grad
loss.backward()

# test3
z1 = torch.randn(8, 128, requires_grad=True)
z2 = torch.randn(8, 128, requires_grad=True)

criterion = NTXent(tau=0.7)
loss_random = criterion(z1, z2)

z3 = torch.randn(8, 128, requires_grad=True)
z4 = z3.clone()
loss_same = criterion(z3, z4)

assert loss_random > loss_same

# test4
dataset = get_train_val_dataset("./data")
loader = load_data(dataset, batch_size=8)

batch = next(iter(loader))
(view1, view2), labels = batch

assert view1.shape == (8, 3, 32, 32)
assert view2.shape == view1.shape

model = SimCLRModel(encoder, projection_head)
z1 = model(view1)
z2 = model(view2)
assert z1.shape == (8, 128)
assert z2.shape == z1.shape

loss = criterion(z1, z2)
assert loss.ndim == 0
assert torch.isfinite(loss)
assert loss.requires_grad

loss.backward()