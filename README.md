# Computer-Vision
# Animal Image Classification

This project focuses on classifying animal images using a Convolutional Neural Network (CNN) implemented in Keras. The dataset used is the 'animals10' dataset from Kaggle.

## Project Structure

- **raw-img/**: Contains the raw animal images for training and validation.
- **drive/MyDrive/main_test_images/**: Contains the test images.
- **kaggle.json**: Kaggle API credentials (ensure this file is present and configured).
- **notebook.ipynb**: The Jupyter Notebook containing the code for data preprocessing, model building, and evaluation.
- **README.md**: This file.

## Steps

1. **Mount Google Drive:** Mount your Google Drive to access the dataset and store the Kaggle API credentials.
2. **Install Kaggle Library:** Install the Kaggle library to download the dataset.
3. **Download Dataset:** Download the 'animals10' dataset using the Kaggle API.
4. **Unzip Dataset:** Unzip the downloaded dataset.
5. **Import Libraries:** Import necessary libraries such as OpenCV, Matplotlib, Keras, etc.
6. **Data Understanding:** Visualize sample images from each class.
7. **Data Preprocessing:**
   - Use ImageDataGenerator to rescale, augment, and split the data into training and validation sets.
8. **Model Building:**
   - Define a CNN model using Keras with appropriate layers (Conv2D, MaxPool2D, Flatten, Dense, Dropout).
   - Compile the model with an optimizer (e.g., Adam) and loss function (e.g., binary crossentropy).
9. **Model Training:**
   - Train the model on the training set and validate it on the validation set.
   - Use EarlyStopping to prevent overfitting.
10. **Model Evaluation:**
    - Evaluate the model on the test set using classification_report and confusion_matrix.

## Dependencies

- Python 3
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- scikit-learn
- keras_drop_block

## Usage

1. Open the `notebook.ipynb` file in Google Colab.
2. Execute the code cells sequentially.
3. Ensure the `kaggle.json` file is in the correct location and configured.
4. Modify the paths to the datasets if necessary.
