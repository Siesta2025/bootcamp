import pytest
import torch

from representation_lab.checkpoints import (
    load_encoder_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from representation_lab.models import ProjectionHead, SimCLRModel, SmallResNet


def make_training_components():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    return model, optimizer, scheduler


def take_training_step(model, optimizer, scheduler):
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()


def assert_model_states_equal(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_training_checkpoint_round_trip(tmp_path):
    source_model, source_optimizer, source_scheduler = make_training_components()
    take_training_step(source_model, source_optimizer, source_scheduler)
    checkpoint_path = tmp_path / "training.pt"

    save_training_checkpoint(
        path=checkpoint_path,
        epoch=3,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        best_metric=0.75,
    )

    target_model, target_optimizer, target_scheduler = make_training_components()
    start_epoch, best_metric = load_training_checkpoint(
        path=checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        device=torch.device("cpu"),
    )

    assert start_epoch == 4
    assert best_metric == pytest.approx(0.75)
    assert_model_states_equal(source_model.state_dict(), target_model.state_dict())
    assert source_optimizer.state_dict()["param_groups"] == (
        target_optimizer.state_dict()["param_groups"]
    )
    source_momentum = next(iter(source_optimizer.state.values()))["momentum_buffer"]
    target_momentum = next(iter(target_optimizer.state.values()))["momentum_buffer"]
    assert torch.equal(source_momentum, target_momentum)
    assert source_scheduler.state_dict() == target_scheduler.state_dict()


@pytest.mark.parametrize(
    ("metric_key", "metric_value"),
    [
        ("best_val_accuracy", 0.8),
        ("best_val_loss", 1.2),
    ],
)
def test_load_training_checkpoint_supports_legacy_metric_keys(
    tmp_path,
    metric_key,
    metric_value,
):
    source_model, source_optimizer, source_scheduler = make_training_components()
    checkpoint = {
        "epoch": 1,
        "model_state_dict": source_model.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
        "scheduler_state_dict": source_scheduler.state_dict(),
        metric_key: metric_value,
    }
    checkpoint_path = tmp_path / f"{metric_key}.pt"
    torch.save(checkpoint, checkpoint_path)
    target_model, target_optimizer, target_scheduler = make_training_components()

    start_epoch, best_metric = load_training_checkpoint(
        path=checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        device=torch.device("cpu"),
    )

    assert start_epoch == 2
    assert best_metric == pytest.approx(metric_value)


def test_load_encoder_checkpoint_extracts_encoder_state(tmp_path):
    source_encoder = SmallResNet()
    source_model = SimCLRModel(
        source_encoder,
        ProjectionHead(in_dim=256, out_dim=128),
    )
    checkpoint_path = tmp_path / "simclr.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)
    target_encoder = SmallResNet()

    returned_encoder = load_encoder_checkpoint(target_encoder, checkpoint_path)

    assert returned_encoder is target_encoder
    assert_model_states_equal(source_encoder.state_dict(), target_encoder.state_dict())
