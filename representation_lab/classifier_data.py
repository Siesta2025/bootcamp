from torchvision import datasets
from torch.utils.data import DataLoader


def get_train_val_dataset(path, transform):
    dataset = datasets.CIFAR10(
        root=path,
        train=True,
        transform=transform,
        download=True,
    )

    return dataset


def get_test_dataset(path, transform):
    dataset = datasets.CIFAR10(
        root=path,
        train=False,
        transform=transform,
        download=True,
    )

    return dataset


def load_data(dataset, shuffle, batch_size=8, num_workers=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader
