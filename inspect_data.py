from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1.
transform = transforms.ToTensor()

# 2.
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

# 3.
image, label = dataset[0]
print(f"Type: {type(image)}")
print(f"Image shape: {image.shape}")
print(f"Image data type: {image.dtype}")
print(f"Image min value: {image.min()}")
print(f"Image max value: {image.max()}")
print(f"Label: {label}")
print(f"Label type: {type(label)}")

# 4. 
dataloader = DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=0,
)

# 5. 
images, labels = next(iter(dataloader))

# 6. 
print(f"Images shape: {images.shape}")
print(f"Images data type: {images.dtype}")
print(f"Labels shape: {labels.shape}")
print(f"Labels data type: {labels.dtype}")
