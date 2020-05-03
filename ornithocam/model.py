import torch


def load_model() -> torch.nn.Module:
    model = torch.hub.load("pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True)
    model.eval()
    return model