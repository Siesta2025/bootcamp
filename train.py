import torch
import numpy as np
import random
import argparse
import csv
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Classifier, SmallResNet

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    train_running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        train_running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_running_loss / train_total
    train_accuracy = train_correct / train_total
    return train_loss, train_accuracy

def evaluate(model, loader, criterion, device):
    model.eval()

    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            test_running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_loss = test_running_loss / test_total
    test_accuracy = test_correct / test_total
    return test_loss, test_accuracy

def save_checkpoint(path, epoch, model, optimizer, best_accuracy, scheduler):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, device, scheduler):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_accuracy = checkpoint["best_accuracy"]
    return start_epoch, best_accuracy

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
        default=5,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="training",
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
    assert (Path("./checkpoints") / run_name).exists() == False, f"Run name '{run_name}' already exists. Please choose a different run name."

    set_seed(seed)

    path = Path("./checkpoints") / run_name
    path.mkdir(parents=True, exist_ok=True)
    
    metric_path = path / "metrics.csv"
    if not metric_path.exists():
        with open(metric_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "lr", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])

    start_epoch = 0
    best_accuracy = 0.0
    last_checkpoint_path = path / "last.pt"
    best_checkpoint_path = path / "best.pt"
    resume_last = False
    resume_best = False
    assert not (resume_last and resume_best), "Cannot resume both last and best checkpoints simultaneously."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    encoder = SmallResNet()
    model = Classifier(encoder).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3, 4],
        gamma=0.1,
    )

    if resume_last:
        start_epoch, best_accuracy = load_checkpoint(last_checkpoint_path, model, optimizer, device, scheduler)
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")
    
    elif resume_best:
        start_epoch, best_accuracy = load_checkpoint(best_checkpoint_path, model, optimizer, device, scheduler)
        print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.4f}")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"epoch: {epoch+1}, training loss: {train_loss:.4f}, training accuracy: {train_accuracy:.4f}")

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"epoch: {epoch+1}, test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(best_checkpoint_path, epoch, model, optimizer, best_accuracy, scheduler)
            print(f"New best accuracy: {best_accuracy:.4f}. Checkpoint saved.")
        save_checkpoint(last_checkpoint_path, epoch, model, optimizer, best_accuracy, scheduler)
        print(f"Checkpoint saved for epoch {epoch+1}.")

        with open(metric_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, current_lr, train_loss, train_accuracy, test_loss, test_accuracy])
