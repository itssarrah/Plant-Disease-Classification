# Plant Disease Classification using ResNet

A deep learning project that classifies plant diseases using a ResNet50 model trained on a dataset of 41 different plant disease categories.

## Project Overview

This project implements a convolutional neural network (ResNet50) to classify plant diseases from leaf images. The model achieves high accuracy in identifying various diseases across 17 different plant species.

### Key Features:
- Utilizes transfer learning with ResNet50 architecture
- Processes and classifies images of diseased plant leaves
- Handles 41 unique disease categories
- Includes comprehensive data preprocessing and visualization
- Achieves high prediction accuracy on test images

## Dataset

The dataset contains:
- 73,237 training images
- 41 unique disease classes
- 17 different plant species
- Balanced distribution across categories

Dataset structure:
combined_dataset/
train/
Apple___Apple_scab/
Apple___Black_rot/
... (41 categories)
valid/
... (same structure as train)


## Implementation

### Data Preprocessing
- Image resizing and squaring (256x256)
- Normalization using ImageNet stats
- Data augmentation
- Class-balanced sampling

### Model Architecture
```python
ResNet50(
  (fc): Sequential(
    (0): Linear(in_features=2048, out_features=512)
    (1): ReLU()
    (2): Dropout(p=0.5)
    (3): Linear(in_features=512, out_features=41)
  )
)
Training
Optimizer: Adam (lr=0.001)

Loss Function: CrossEntropyLoss

Batch Size: 32

Epochs: 10

Hardware: CUDA-enabled GPU
```

###  Results
The model achieved:

- High training accuracy
- Excellent validation performance
- Perfect prediction on test images

Sample predictions:

```
Label: test1.jpg , Predicted: Apple___Apple_scab
Label: test2.jpg , Predicted: Tomato___healthy
...
