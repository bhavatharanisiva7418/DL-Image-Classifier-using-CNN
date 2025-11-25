# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Import Required Libraries
- PyTorch for building and training the model
- torchvision for loading MNIST dataset
- matplotlib for visualization

### STEP 2: Load and Preprocess the Dataset
- Normalize pixel values to [0, 1]
- Split into training and test sets
- Use DataLoader for batching

### STEP 3: Build the CNN Model
- Define convolutional, pooling, and fully connected layers
- Implement forward pass using ReLU activations

### STEP 4: Define Loss Function and Optimizer
- Use `CrossEntropyLoss` for multi-class classification
- Use `Adam` optimizer for faster convergence

### STEP 5: Train the Model
- Loop through epochs and batches
- Compute loss and update weights
- Track training loss

### STEP 6: Evaluate Model Performance
- Test on unseen data
- Plot **confusion matrix** and compute **classification report**
- Predict on new sample images


### RESULT
The CNN model successfully classified the MNIST handwritten digits with high accuracy (~99%). The training loss decreased steadily over epochs, the confusion matrix shows correct predictions for almost all digits, and the model can correctly predict new unseen samples.

