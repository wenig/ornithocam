import fire
from ornithocam.detect import detect as d
from ornithocam.webcam import webcam_detect as wd


class Tool:
    @staticmethod
    def detect(filename: str, from_url: bool, top_n: int = 20):
        d(filename, from_url, top_n)

    @staticmethod
    def webcam(record: bool, if_bird: bool):
        wd(record, if_bird)


if __name__ == "__main__":
    fire.Fire(Tool)