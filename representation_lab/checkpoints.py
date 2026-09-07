import torch


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