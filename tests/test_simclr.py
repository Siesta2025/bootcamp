import pytest
import torch

from representation_lab.models import SmallResNet, ProjectionHead, SimCLRModel
from representation_lab.simclr_data import get_train_val_dataset, load_data
from representation_lab.losses import NTXent


def test_encoder_returns_256_features():
    x = torch.randn(8, 3, 32, 32)
    encoder = SmallResNet()
    h = encoder(x)
    assert h.shape == (8, 256)


def test_projection_head_returns_128_features():
    h = torch.randn(8, 256)
    projection_head = ProjectionHead(256, 128)
    z = projection_head(h)
    assert z.shape == (8, 128)


def test_ntxent_returns_finite_scalar_and_propagates_gradients():
    z1 = torch.randn(8, 128, requires_grad=True)
    z2 = torch.randn(8, 128, requires_grad=True)

    criterion = NTXent(tau=0.7)
    loss = criterion(z1, z2)
    assert loss.shape == torch.Size([])  # scalar value
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert z1.grad is not None
    assert z2.grad is not None
    assert torch.isfinite(z1.grad).all()
    assert torch.isfinite(z2.grad).all()


def test_ntxent_is_lower_for_matching_pairs():
    torch.manual_seed(0)

    z1 = torch.randn(8, 128, requires_grad=True)
    random_z2 = torch.randn(8, 128, requires_grad=True)
    matching_z2 = z1.clone()

    criterion = NTXent(tau=0.7)

    random_loss = criterion(z1, random_z2)
    matching_loss = criterion(z1, matching_z2)

    assert random_loss > matching_loss


@pytest.mark.integration
def test_simclr_pipeline_on_cifar_batch():
    dataset = get_train_val_dataset("./data")
    loader = load_data(dataset, batch_size=8)

    batch = next(iter(loader))
    (view1, view2), _ = batch

    assert view1.shape == (8, 3, 32, 32)
    assert view2.shape == view1.shape

    criterion = NTXent(tau=0.7)
    encoder = SmallResNet()
    projection_head = ProjectionHead(256, 128)

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
