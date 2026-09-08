import torch


def save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_metric,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(checkpoint, path)


def load_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_metric = _read_best_metric(checkpoint)
    return start_epoch, best_metric


def load_encoder_checkpoint(encoder, path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    encoder_state_dict = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }

    encoder.load_state_dict(encoder_state_dict)
    return encoder


def _read_best_metric(checkpoint):
    for key in ("best_metric", "best_val_accuracy", "best_val_loss"):
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        "Checkpoint does not contain 'best_metric' or a supported legacy metric key."
    )
