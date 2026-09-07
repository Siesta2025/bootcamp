import torch

def train_simclr_one_epoch(
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


def evaluate_simclr(
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