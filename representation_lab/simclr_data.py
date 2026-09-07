from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2

def get_train_val_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    two_view_transform = TwoViewTransform(transform)

    dataset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=two_view_transform,
        download=True,
    )

    return dataset

def get_test_dataset(dataset_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    two_view_transform = TwoViewTransform(transform)

    dataset = datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=two_view_transform,
        download=True,
    )

    return dataset

def load_data(dataset, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return loader
