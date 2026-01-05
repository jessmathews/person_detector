# Person Detector


# Pre-Requisites
Complete these first:
- [ ] Python3
- [ ] Download YOLO Model weights
  - [ ] [Download YOLOV11 nano model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
  - [ ] [Download YOLOV11 small model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
---


# Setting up 
```sh
git clone https://github.com/jessmathews/person_detector.git

cd person_detector

# create a virtual environment to work in
python3 -m venv env
source env/bin/activate

# install requirements
pip install -r requirements.txt

```

# Usage 
## Live webcam
```sh
# run with live webcam and confidence of >90% 
python3 main.py -w  -c 0.9
```

## Recorded Video 
```sh
#run on recorded video footage with confidence score > 90%
python3 main.py -v /path/to/video.mp4  -c 0.9

```
