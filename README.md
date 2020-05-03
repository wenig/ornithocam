# Ornithocam

## Installation
```
pip install -r requirements.txt
python setup.py install
```

## Example (from webcam)
```python
from ornithocam.webcam import webcam_detect

webcam_detect()
```

## Example (from terminal)
```shell script
python ornithocam_tool.py webcam --record=True --if_bird=True
```
