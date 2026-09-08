import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def classification_criterion():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def tiny_classification_loader():
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(8, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (8,), generator=generator)
    dataset = TensorDataset(images, labels)

    return DataLoader(dataset, batch_size=4, shuffle=False)
