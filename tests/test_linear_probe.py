import math

import pytest
import torch

from representation_lab.linear_probe import LinearProbe
from representation_lab.models import SmallResNet
from representation_lab.training import evaluate_classifier, train_linear_probe_one_epoch


@pytest.fixture
def linear_probe(device):
    return LinearProbe(SmallResNet(), num_classes=10).to(device)


@pytest.fixture
def linear_probe_optimizer(linear_probe):
    return torch.optim.SGD(
        linear_probe.classifier.parameters(),
        lr=0.1,
        momentum=0.9,
    )


def test_linear_probe_freezes_encoder(linear_probe):
    assert all(
        not parameter.requires_grad
        for parameter in linear_probe.frozen_encoder.parameters()
    )


def test_linear_probe_keeps_classifier_trainable(linear_probe):
    assert all(
        parameter.requires_grad
        for parameter in linear_probe.classifier.parameters()
    )


def test_linear_probe_returns_class_logits(linear_probe, device):
    images = torch.randn(8, 3, 32, 32, device=device)

    logits = linear_probe(images)

    assert logits.shape == (8, 10)


def test_backward_populates_only_classifier_gradients(
    linear_probe,
    classification_criterion,
    device,
):
    images = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (8,), device=device)

    loss = classification_criterion(linear_probe(images), labels)
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in linear_probe.frozen_encoder.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in linear_probe.classifier.parameters()
    )


def test_optimizer_updates_only_classifier(
    linear_probe,
    linear_probe_optimizer,
    classification_criterion,
    device,
):
    images = torch.randn(8, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (8,), device=device)
    encoder_before = next(linear_probe.frozen_encoder.parameters()).detach().clone()
    classifier_before = next(linear_probe.classifier.parameters()).detach().clone()

    linear_probe_optimizer.zero_grad()
    loss = classification_criterion(linear_probe(images), labels)
    loss.backward()
    linear_probe_optimizer.step()

    encoder_after = next(linear_probe.frozen_encoder.parameters()).detach().clone()
    classifier_after = next(linear_probe.classifier.parameters()).detach().clone()
    assert torch.allclose(encoder_before, encoder_after)
    assert not torch.allclose(classifier_before, classifier_after)


def test_linear_probe_training_returns_valid_metrics_and_modes(
    linear_probe,
    linear_probe_optimizer,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    loss, accuracy = train_linear_probe_one_epoch(
        linear_probe,
        tiny_classification_loader,
        classification_criterion,
        linear_probe_optimizer,
        device,
    )

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert not linear_probe.frozen_encoder.training
    assert linear_probe.classifier.training


def test_classifier_evaluation_does_not_update_probe_state(
    linear_probe,
    classification_criterion,
    tiny_classification_loader,
    device,
):
    before = {
        key: value.detach().clone()
        for key, value in linear_probe.state_dict().items()
    }

    loss, accuracy = evaluate_classifier(
        linear_probe,
        tiny_classification_loader,
        classification_criterion,
        device,
    )

    after = linear_probe.state_dict()
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert all(torch.equal(before[key], after[key]) for key in before)
