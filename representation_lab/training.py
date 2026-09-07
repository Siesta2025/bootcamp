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

def train_supervised_one_epoch(model, loader, criterion, optimizer, device):
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

def evaluate_classifier(model, loader, criterion, device):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy

def train_linear_probe_one_epoch(prober, loader, optimizer, criterion, device):
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