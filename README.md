# Deepfake Detector ( for Intelligent Model Design Using AI course)
This project is an advanced computer vision framework designed to tackle the growing challenge of synthetic media and facial manipulations. Developed as a centerpiece for the Intelligent Model Design course, this project leverages a high-performance EfficientNet architecture to identify subtle spatial inconsistencies and artifacts that characterize AI-generated deepfakes. By optimizing the detection pipeline for local execution on Apple Silicon using TensorFlow-Metal, the model achieves high-precision results while remaining accessible for real-time forensics. This repository represents a transition from conceptual AI theory to a practical, explainable solution for verifying digital authenticity.
## Acknowledgements

 - [This project is based on the architecture provided here.](https://github.com/umitkacar/multimodal-deepfake-detector)

## Deployment

To deploy this project run: 

##1 Installation
```bash
  # Clone the repository
git clone https://github.com/umitkacar/DeepFake-EfficientNet.git
cd DeepFake-EfficientNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
##2 Extraction of Images/Video Frames
```bash
# Extract faces from videos
python scripts/extract_faces.py \
    --input-dir /path/to/videos \
    --output-dir /path/to/extracted_faces \
    --mode video \
    --batch-size 60 \
    --frame-skip 30

# Extract faces from images
python scripts/extract_faces.py \
    --input-dir /path/to/images \
    --output-dir /path/to/extracted_faces \
    --mode image
```
##3 Training
```bash
python scripts/train.py \
    --train-real /path/to/train/real \
    --train-fake /path/to/train/fake \
    --val-real /path/to/val/real \
    --val-fake /path/to/val/fake \
    --output-dir outputs \
    --batch-size 32 \
    --epochs 20 \
    --lr 8e-4 \
    --model efficientnet-b1
```
##4 Testing
```bash
python scripts/test.py \
    --test-real /path/to/test/real \
    --test-fake /path/to/test/fake \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output-dir test_results \
    --batch-size 100 \
    --save-predictions
```
##5 Inputting on New Image/Video frames
```bash
# Single image inference
python scripts/inference.py \
    --input /path/to/image.jpg \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model efficientnet-b1

# Batch inference on directory
python scripts/inference.py \
    --input /path/to/images/directory \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model efficientnet-b1 \
    --output predictions.csv
```
##6 Running the Web App
```bash
#Web app (provides a local host URL)
pyhton scripts/webapp.py
```



