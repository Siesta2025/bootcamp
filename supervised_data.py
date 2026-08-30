from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

image, label = dataset[0]
print(f"Type: {type(image)}")
print(f"Image shape: {image.shape}")
print(f"Image data type: {image.dtype}")
print(f"Image min value: {image.min()}")
print(f"Image max value: {image.max()}")
print(f"Label: {label}")
print(f"Label type: {type(label)}")

dataloader = DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=0,
)

images, labels = next(iter(dataloader))

print(f"Images shape: {images.shape}")
print(f"Images data type: {images.dtype}")
print(f"Labels shape: {labels.shape}")
print(f"Labels data type: {labels.dtype}")
