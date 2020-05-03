import fire
import urllib
from model import load_model
from utils import infer_image, infer_numpy


def download_image(url):
    filename = "tempimg.jpg"
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    return filename


def detect(filename: str, from_url: bool = False, top_n: int = 20):
    if from_url:
        filename = download_image(filename)
    model = load_model()
    rankings = infer_image(model, filename)
    for name, conf in rankings[:top_n]:
        print(f"{conf*100:10.2f}\t{name}")


def detect_from_frame(model, frame, top_n: int = 20):
    return infer_numpy(model, frame)


if __name__ == "__main__":
    fire.Fire(detect)
