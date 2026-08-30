import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

def load_data(dataset):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    two_view_transform = TwoViewTransform(transform)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    return loader

if __name__ == "__main__":
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=two_view_transform,
        download=True,
    )

    loader = load_data(dataset)
