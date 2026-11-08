# Volleyball Vision Analytics: Object Detection & Tracking

This module is responsible for the first stage of the Volleyball Vision Analytics pipeline. Its primary goal is to detect and track the volleyball across 2D video frames using a custom-trained YOLO model. 

The output is a time-series dataset of (X, Y) coordinates, which is subsequently used for rally boundary classification and attack speed estimation.

## Pipeline Overview
1. **Frame Extraction:** Extracting frames from raw volleyball match footage (`.mp4`) for manual annotation.
2. **Model Training:** Training a YOLO-based object detection model to accurately recognize the volleyball, handling challenges like motion blur and occlusion.
3. **Ball Tracking:** Running inference on full match clips to extract frame-by-frame (X, Y) coordinates and saving the results to a CSV file.

## Folder Structure
* `model_training/`: Contains Python scripts for data preparation, YOLO training, and inference.
* `models/weights/`: Stores the trained YOLO model weights (`.pt` files).
* `notebooks/`: Jupyter notebooks for quick experiments and model validation.

## Setup
Ensure you have the main project data folder populated at the root level (`../data/`). 

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
*Scripts should be executed in numerical order from the `model_training` directory.*