# Beyond Single-Centroid: Modeling Multi-Modal Forgery Manifold for Generalizable Face Forgery Detection
The official PyTorch implementation for the following paper:Beyond Single-Centroid: Modeling Multi-Modal Forgery Manifold for Generalizable Face Forgery Detection
## Setup

### 1. Dataset

Download datasets and place them in `./data/` folder.  
For example, download [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) and place it:

```text
.
└── data
    └── Celeb-DF-v2
        ├── Celeb-real
        │   └── videos
        │       └── *.mp4
        ├── Celeb-synthesis
        │   └── videos
        │       └── *.mp4
        ├── Youtube-real
        │   └── videos
        │       └── *.mp4
        └── List_of_testing_videos.txt
```

For other datasets, please refer to `./data/datasets.md`.
### 2. Requirements
Install the required packages with:
```text
pip install -r requirements.txt
```
### 3. Model

Download **CLIP-ViT-L/14** and place the model files in the `./src/hf_home/hub` folder.

Alternatively, you can modify the model path in `model_clip.py` by changing:

```python
HUB_REPO_DIR = "/root/autodl-tmp/src/hf_home/hub/models--openai--clip-vit-large-patch14"
```
## Training

1. Download [FF++](https://github.com/ondyari/FaceForensics) real videos and place them in the `./data/` folder:

```text
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── train.json
        ├── val.json
        └── test.json
```
2. Download the landmark detector (`shape_predictor_81_face_landmarks.dat`) from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in `./src/preprocess/`.
3. Run the two codes to extractvideo frames, landmarks, and bounding boxes:
```python
python3 src/preprocess/crop_dlib_ff.py -d Original -c c23
python3 src/preprocess/crop_retina_ff.py -d Original -c c23
```
4.download code for landmark augmentation:
```python
mkdir src/utils/library
git clone https://github.com/AlgoHunt/Face-Xray.git src/utils/library
```
This step is important for landmark augmentation. The code can still run without it, but the performance may drop significantly. Please ensure that this step is completed correctly.

5.Run the training:
```python
python3 src/train.py src/configs/clip/base.json -n clip_base
```
## Test
For example, run the inference on Celeb-DF-v2:
```python
python3 src/inference/inference.py -w /path/to/your/trained_model.tar -d DFD -n 32 --clip_snapshot_dir /path/to/your/clip_model
```
