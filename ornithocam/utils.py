import torch
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet


file_path = os.path.dirname(os.path.abspath(__file__))


def get_class_name(ranks: torch.Tensor):
    CLASSNAMES = "imagenet1000_clsidx_to_labels.json"
    classes = []
    with open(os.path.join(file_path, CLASSNAMES), "r") as f:
        for line in f.readlines():
            raw = line.strip("{").strip("}")
            key, value = raw.split(": ")
            classes.append(value.strip("'").strip("',\n"))
    sorted_ranks = ranks.argsort(descending=True)
    return np.array(classes)[sorted_ranks]


def numpy_to_image(array: np.ndarray) -> Image:
    return Image.fromarray(array)


def preprocess_mobilenet(input_image: Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def infer_image(model: torch.nn.Module, filename: str):
    input_image = Image.open(filename)
    input_batch = preprocess_mobilenet(input_image)
    return infer_tensor(model, input_batch)


def infer_tensor(model: torch.nn.Module, input_batch: torch.Tensor):
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    confidence = torch.nn.functional.softmax(output[0], dim=0).flatten()
    class_names = get_class_name(confidence)
    return list(zip(class_names, confidence.sort(descending=True)[0]))


def infer_numpy(model: torch.nn.Module, input_numpy: np.ndarray):
    input_image = numpy_to_image(input_numpy)
    input_batch = preprocess_mobilenet(input_image)
    return infer_tensor(model, input_batch)


def get_bird_keywords() -> []:
    return []
