import math

import pytest
import torch
from torch.utils.data import Subset
from torchvision import transforms

from representation_lab.classification_data import (
    get_test_dataset,
    get_train_val_dataset,
    load_data,
)
from representation_lab.models import Classifier, SmallResNet
from representation_lab.training import evaluate_classifier, train_supervised_one_epoch


@pytest.fixture
def classifier(device):
    return Classifier(SmallResNet()).to(device)


def assert_valid_classification_metrics(loss, accuracy):
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0


def test_classifier_returns_class_logits(classifier, device):
    images = torch.randn(8, 3, 32, 32, device=device)

    logits = classifier(images)

    assert logits.shape == (8, 10)


def test_supervised_training_returns_valid_metrics(
    classifier,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    loss, accuracy = train_supervised_one_epoch(
        classifier,
        tiny_classification_loader,
        classification_criterion,
        optimizer,
        device,
    )

    assert_valid_classification_metrics(loss, accuracy)


def test_supervised_training_updates_parameters(
    classifier,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    before = next(classifier.parameters()).detach().clone()

    train_supervised_one_epoch(
        classifier,
        tiny_classification_loader,
        classification_criterion,
        optimizer,
        device,
    )

    after = next(classifier.parameters()).detach().clone()
    assert not torch.allclose(before, after)


def test_evaluate_classifier_returns_valid_metrics(
    classifier,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    loss, accuracy = evaluate_classifier(
        classifier,
        tiny_classification_loader,
        classification_criterion,
        device,
    )

    assert_valid_classification_metrics(loss, accuracy)


def test_evaluate_classifier_does_not_update_parameters(
    classifier,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    before = next(classifier.parameters()).detach().clone()

    evaluate_classifier(
        classifier,
        tiny_classification_loader,
        classification_criterion,
        device,
    )

    after = next(classifier.parameters()).detach().clone()
    assert torch.allclose(before, after)


@pytest.mark.integration
def test_supervised_pipeline_on_cifar_subsets(
    classifier,
    classification_criterion,
    device,
):
    transform = transforms.ToTensor()
    full_train_dataset = get_train_val_dataset("data", transform=transform)
    full_test_dataset = get_test_dataset("data", transform=transform)
    train_dataset = Subset(full_train_dataset, range(8))
    val_dataset = Subset(full_train_dataset, range(8, 16))
    test_dataset = Subset(full_test_dataset, range(8))

    train_loader = load_data(train_dataset, shuffle=False, batch_size=4)
    val_loader = load_data(val_dataset, shuffle=False, batch_size=4)
    test_loader = load_data(test_dataset, shuffle=False, batch_size=4)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)

    train_metrics = train_supervised_one_epoch(
        classifier,
        train_loader,
        classification_criterion,
        optimizer,
        device,
    )
    val_metrics = evaluate_classifier(
        classifier,
        val_loader,
        classification_criterion,
        device,
    )
    test_metrics = evaluate_classifier(
        classifier,
        test_loader,
        classification_criterion,
        device,
    )

    for loss, accuracy in (train_metrics, val_metrics, test_metrics):
        assert_valid_classification_metrics(loss, accuracy)
