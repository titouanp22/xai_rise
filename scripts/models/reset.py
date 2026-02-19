import torch
from torchvision import models
import torchvision.models as models


def get_resnet50(device = "cpu"):
    model = models.resnet50(pretrained=True)
    model.eval()

    # weights = models.ResNet50_Weights.IMAGENET1K_V2
    # model = models.resnet50(weights=weights).eval().to(device)
    return model #, weights


@torch.no_grad()
def predict_class(model, input_tensor):
    logits = model(input_tensor)
    return int(logits.argmax(dim=1).item())

