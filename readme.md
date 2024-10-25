
# Dog-Cat Detection Project

This project builds an object detection model to classify images of dogs and cats using YOLO11.

## Table of Contents

- [Introduction](#introduction)
- [Project Workflow](#project-workflow)
  - [1. Data Collection and Annotation](#1-data-collection-and-annotation)
  - [2. Dataset Preparation in Ultralytics](#2-dataset-preparation-in-ultralytics)
  - [3. Model Training in Google Colab](#3-model-training-in-google-colab)
  - [4. Testing and Evaluation](#4-testing-and-evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Introduction

The purpose of this project is to create a YOLO-based model that distinguishes between images of dogs and cats.

## Project Workflow

### 1. Data Collection and Annotation

- **Image Upload**: Collected images were uploaded to [Roboflow](https://roboflow.com/).
- **Annotation**: Each image was manually annotated in Roboflow to label dogs and cats.

### 2. Dataset Preparation in Ultralytics

- **Dataset Export**: The annotated dataset was exported in YOLO format.
- **Data Import**: This dataset was prepared for use in Ultralytics YOLO11.

### 3. Model Training in Google Colab

- **Model Setup**: YOLO11 from Ultralytics was configured on Google Colab.
- **Training**: The model was trained with optimized parameters to improve detection accuracy.
- **Testing**: Used test images to evaluate the model, displaying bounding boxes and confidence scores.

### 4. Testing and Evaluation

- Additional test images were loaded to assess the model’s accuracy in detecting and classifying dogs and cats.

## Results

The trained YOLO11 model achieved reliable accuracy on the test set.

## Requirements

- [Python](https://www.python.org/)
- [Google Colab](https://colab.research.google.com/)
- Ultralytics YOLO11
- Roboflow (for annotation)

## Acknowledgments

Thanks to Roboflow, Ultralytics, and Google Colab for enabling this project.

