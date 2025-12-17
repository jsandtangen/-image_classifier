# Cat and Dog Image Classifier

## Overview
This project is a convolutional neural network (CNN) built to classify images of cats and dogs using **TensorFlow 2.0** and **Keras**.  
The project is designed to be run in **Google Colaboratory** and is part of a machine learning challenge focused on practical image classification.

The goal is to achieve at least **63% classification accuracy** on unseen test images (with bonus for 70%+).

---

## Project Goals
- Build an end-to-end image classification pipeline
- Use image generators to load and preprocess image data
- Apply data augmentation to reduce overfitting
- Train and evaluate a CNN on real image data
- Visualize model performance and predictions

---

## Dataset Structure
The dataset is downloaded automatically in the notebook and has the following structure:

cats_and_dogs/
│
├── train/
│ ├── cats/ # Cat images for training
│ └── dogs/ # Dog images for training
│
├── validation/
│ ├── cats/ # Cat images for validation
│ └── dogs/ # Dog images for validation
│
└── test/ # Unlabeled test images


- Training set: 2000 images  
- Validation set: 1000 images  
- Test set: 50 images (no labels)

---

## Technologies Used
- Python
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Google Colab

---

## Model Description
The model is built using the **Keras Sequential API** and consists of:
- Multiple `Conv2D` layers for feature extraction
- `MaxPooling2D` layers for downsampling
- A fully connected (Dense) layer with ReLU activation
- A final output layer producing class probabilities

The model is compiled with:
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Metric: Accuracy

---

## Training Strategy
- Images are rescaled from `[0, 255]` to `[0, 1]`
- Data augmentation is applied to the training set using random transformations
- The model is trained using generators for efficiency
- Accuracy and loss are tracked for both training and validation data

---

## Results
After training, the model:
- Plots training and validation accuracy/loss
- Predicts whether each test image is a cat or a dog
- Displays predictions visually with confidence percentages

---

## How to Run
1. Open the notebook in Google Colab
2. Run the cells in order (Cell 1 → Cell 11)
3. Adjust epochs or batch size if desired
4. Verify final accuracy in the last cell

---

## Notes
- The test dataset is not shuffled to ensure correct prediction order
- Data augmentation is critical due to limited training data
- Accuracy can vary slightly between runs

---

## Status
✅ Completed  
📈 Further improvements possible with more data or tuning

---

## Author
This project was created as part of a machine learning certification challenge and serves as a learning and portfolio project.


