import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import SimCLRModel, ProjectionHead, SmallResNet
from simclr_data import get_train_val_dataset


class NTXent(nn.Module):
    def __init__(self, tau=0.7):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2):
        assert z1.shape == z2.shape, \
            "Two outputs must have the same shape."

        batch_size = z1.size(0)

        z = torch.cat([z1, z2], dim=0)

        z = F.normalize(z, dim=1)

        logits = z @ z.T
        logits = logits / self.tau

        self_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device,
        )
        logits = logits.masked_fill(
            self_mask,
            float("-inf"),
        )


        targets = (
            torch.arange(
                2 * batch_size,
                device=z.device,
            )
            + batch_size
        ) % (2 * batch_size)

        return F.cross_entropy(logits, targets)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    total = 0

    for (view1, view2), _ in loader:
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()

        z1 = model(view1)
        z2 = model(view2)

        loss = criterion(z1, z2)

        loss.backward()
        optimizer.step()

        batch_size = view1.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / total


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    total = 0

    with torch.inference_mode():
        for (view1, view2), _ in loader:
            view1 = view1.to(device)
            view2 = view2.to(device)

            z1 = model(view1)
            z2 = model(view2)

            loss = criterion(z1, z2)

            batch_size = view1.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

    return running_loss / total


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    return start_epoch, best_val_loss


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
        default="simclr",
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
        "--tau",
        type=float,
        default=0.7,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    path = Path("./outputs") / args.run_name

    if args.resume is None:
        assert not path.exists(), (
            f"Run '{args.run_name}' already exists."
        )
    else:
        assert path.exists(), (
            f"Run '{args.run_name}' does not exist."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------
    # Data
    # -------------------------

    generator = torch.Generator().manual_seed(
        args.seed
    )

    full_dataset = get_train_val_dataset(
        "./data"
    )

    indices = torch.randperm(
        len(full_dataset),
        generator=generator,
    )

    train_indices = indices[:45000]
    val_indices = indices[45000:]

    train_dataset = Subset(
        full_dataset,
        train_indices,
    )

    val_dataset = Subset(
        full_dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------

    encoder = SmallResNet()

    projector = ProjectionHead(
        in_dim=256,
        out_dim=128,
    )

    model = SimCLRModel(
        encoder,
        projector,
    ).to(device)

    criterion = NTXent(
        tau=args.tau
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )

    # -------------------------
    # Run state
    # -------------------------

    last_checkpoint_path = path / "last.pt"
    best_checkpoint_path = path / "best.pt"
    metric_path = path / "metrics.csv"

    if args.resume is None:
        path.mkdir(parents=True)

        start_epoch = 0
        best_val_loss = float("inf")

        config = vars(args).copy()
        config["device"] = str(device)
        config["dataset"] = "CIFAR10"
        config["model"] = "SmallResNet+ProjectionHead"
        config["projection_dim"] = 128
        config["criterion"] = "NTXent"
        config["optimizer"] = "SGD"
        config["scheduler"] = "MultiStepLR"
        config["train_size"] = 45000
        config["val_size"] = 5000

        with open(
            path / "config.json",
            "w",
        ) as f:
            json.dump(
                config,
                f,
                indent=4,
            )

        with open(
            metric_path,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "lr",
                "train_loss",
                "val_loss",
            ])

    else:
        checkpoint_path = (
            last_checkpoint_path
            if args.resume == "last"
            else best_checkpoint_path
        )

        assert checkpoint_path.exists(), (
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )

        print(
            f"Resuming from epoch {start_epoch}, "
            f"best val loss = {best_val_loss:.4f}"
        )

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(
        start_epoch,
        args.num_epochs,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"epoch: {epoch + 1}, "
            f"train loss: {train_loss:.4f}, "
            f"val loss: {val_loss:.4f}, "
            f"lr: {current_lr:.6f}"
        )

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
            )

            print(
                f"New best val loss: "
                f"{best_val_loss:.4f}"
            )

        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
        )

        with open(
            metric_path,
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_lr,
                train_loss,
                val_loss,
            ])

    print(
        "SimCLR pretraining finished. "
        f"Best validation loss: {best_val_loss:.4f}"
    )