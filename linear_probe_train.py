import torch
import numpy as np
import random
import torch.nn as nn
import argparse
import csv
import json
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from linear_probe import LinearProbe
from models import SmallResNet

def load_encoder_checkpoint(encoder, path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    state_dict = checkpoint["model_state_dict"]

    encoder_dict = {}

    for key in state_dict.keys():
        if key.startswith("encoder."):
            encoder_dict[key[len("encoder."):]] = state_dict[key]

    encoder.load_state_dict(encoder_dict)
    return encoder

def initialize_linear_probe(encoder, num_classes=10):
    prober = LinearProbe(encoder, num_classes=num_classes)
    for param in prober.frozen_encoder.parameters():
        assert param.requires_grad is False
    for param in prober.classifier.parameters():
        assert param.requires_grad is True
    return prober

def get_train_val_dataset(path):
    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=path,
        train=True,
        transform=transform,
        download=True,
    )
    return dataset

def get_test_dataset(path):
    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=path,
        train=False,
        transform=transform,
        download=True,
    )
    return dataset

def load_data(dataset, shuffle, batch_size=8, num_workers=0):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

def train_one_epoch(prober, loader, optimizer, criterion, device):
    prober.to(device)
    prober.frozen_encoder.eval()
    prober.classifier.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for x, label in loader:
        x = x.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = prober(x)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (output.argmax(dim=1) == label).sum().item()

    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy

def evaluate(prober, loader, criterion, device):
    prober.to(device)
    
    prober.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    with torch.inference_mode():
        for x, label in loader:
            x = x.to(device)
            label = label.to(device)

            output = prober(x)

            loss = criterion(output, label)

            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (output.argmax(dim=1) == label).sum().item()

    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy

def save_checkpoint(path, epoch, model, optimizer, best_val_accuracy, scheduler):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, device, scheduler):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_accuracy = checkpoint["best_val_accuracy"]
    return start_epoch, best_val_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        choices=["last", "best"],
        default=None,
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[20, 25],
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        choices=["random", "simclr", "supervised"],
        default="simclr",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers
    seed = args.seed
    lr = args.lr
    momentum = args.momentum
    num_epochs = args.num_epochs
    run_name = args.run_name
    resume = args.resume
    milestones = args.milestones
    gamma = args.gamma
    num_classes = args.num_classes
    encoder_name = args.encoder_name

    path = Path("./outputs") / run_name

    if resume is None:
        assert not path.exists(), f"Run name '{run_name}' already exists. Please choose a different run name."

    elif resume == "last":
        assert path.exists(), f"Run name '{run_name}' does not exist. Please choose a valid run name to resume training."
    
    else:
        assert path.exists(), f"Run name '{run_name}' does not exist. Please choose a valid run name to resume training."
    
    set_seed(seed)

    generator = torch.Generator().manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_val_dataset = get_train_val_dataset("./data")

    test_dataset = get_test_dataset("./data")

    indices = torch.randperm(
        50000,
        generator=generator,
    )
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
        batch_size=batch_size,
        num_workers=num_workers,
    )

    val_loader = load_data(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = load_data(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    encoder = SmallResNet()
    if encoder_name == "simclr":
        load_encoder_checkpoint(encoder, "./outputs/simclr_30/best.pt")
    elif encoder_name == "supervised":
        load_encoder_checkpoint(encoder, "./outputs/supervised_30/best.pt")
    model = initialize_linear_probe(encoder, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=lr,
        momentum=momentum,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )

    last_checkpoint_path = path / "last.pt"
    best_checkpoint_path = path / "best.pt"

    if resume is None:
        path.mkdir(parents=True)

        start_epoch = 0
        best_val_accuracy = 0.0

        config = vars(args).copy()
        config["device"] = str(device)
        config["dataset"] = "CIFAR10"
        config["criterion"] = "CrossEntropyLoss"
        config["optimizer"] = "SGD"
        config["scheduler"] = "MultiStepLR"

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

    elif resume == "last":        
        start_epoch, best_val_accuracy = load_checkpoint(last_checkpoint_path, model, optimizer, device, scheduler)
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_val_accuracy:.4f}")

    else:
        start_epoch, best_val_accuracy = load_checkpoint(best_checkpoint_path, model, optimizer, device, scheduler)
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_val_accuracy:.4f}")

    metric_path = path / "metrics.csv"
    if not metric_path.exists():
        with open(metric_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"epoch: {epoch+1}, training loss: {train_loss:.4f}, training accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f"epoch: {epoch+1}, validation loss: {val_loss:.4f}, validation accuracy: {val_accuracy:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(best_checkpoint_path, epoch, model, optimizer, best_val_accuracy, scheduler)
            print(f"New best accuracy: {best_val_accuracy:.4f}. Checkpoint saved.")

        save_checkpoint(last_checkpoint_path, epoch, model, optimizer, best_val_accuracy, scheduler)
        print(f"Checkpoint saved for epoch {epoch+1}.")

        with open(metric_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, current_lr, train_loss, train_accuracy, val_loss, val_accuracy])

    load_checkpoint(best_checkpoint_path, model, optimizer, device, scheduler)
    print(f"Loading best checkpoint for formal testing.")

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"test loss: {test_loss:.4f}, test_accuracy: {test_accuracy:.4f}.")

