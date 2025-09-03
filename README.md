# Cracking-ML-PUF-with-Linear-ML-Model


This project implements a logistic regression classifier with custom feature mapping for a binary classification task. The code is written in Python and provided as a Jupyter Notebook (assignment_1ipynb.ipynb). It processes input data, maps it into a high-dimensional feature space, trains a logistic regression model, evaluates its performance, and includes a function to decode the model's weights.
Features

Data Loading: Reads input data from text files (public_trn.txt and public_tst.txt) and separates features and labels.
Feature Mapping: Transforms 8-dimensional binary input features into a 121-dimensional feature space, including:
Bias term
Linear features (mapped from 0→1, 1→-1)
Cumulative product features
Quadratic interaction terms


Model Training: Uses scikit-learn's LogisticRegression with feature scaling (StandardScaler) for training.
Evaluation: Computes and displays:
Accuracy, precision, recall, and F1-score
Confusion matrix
Detailed classification report
Training time


Weight Decoding: Includes a function to decode the model's weights into four 64-dimensional vectors (partially implemented).

Requirements
To run the notebook, install the required Python packages. The following dependencies are used:

Python 3.x
numpy
pandas
scikit-learn
time
itertools






Ensure the required data files (public_trn.txt and public_tst.txt) are placed in the project directory.
Install dependencies (see Requirements section).
Open the Jupyter Notebook:jupyter notebook assignment_1ipynb.ipynb



Usage

Data Files:

Place public_trn.txt (training data) and public_tst.txt (test data) in the project directory.
Each file should contain rows of 9 integers (8 features + 1 label), with values 0 or 1, separated by spaces.


Running the Notebook:

Execute the cells in assignment_1ipynb.ipynb sequentially.
The notebook will:
Load and preprocess the data.
Map features using the my_map function.
Train the logistic regression model using my_fit.
Display evaluation metrics and the confusion matrix.
Print the final test accuracy.




Key Functions:

load(file): Loads data from a text file and returns features and labels.
my_map(x_raw): Maps 8-dimensional input features to 121-dimensional features.
my_fit(X_train, y_train, X_test, y_test): Trains the model and evaluates performance.
my_decode(w): Decodes the model's weights (note: r and s vectors are placeholders and may need further implementation).


Output:

The notebook prints:
Training time
Accuracy, precision, recall, and F1-score
Confusion matrix
Classification report
Final test accuracy


Example output:Loading data...
Mapping features...
Train shape: (6400, 121), Test shape: (1600, 121)
Training Time: 0.0839 sec
Accuracy:  0.9263
Precision: 0.8829
Recall:    0.8829
F1-Score:  0.8829
Confusion Matrix:
          Pred 0 | Pred 1
Actual 0 |  1037   |   59   
Actual 1 |   59    |   445  
Classification Report:
              precision    recall  f1-score   support
Class 0         0.95      0.95      0.95      1096
Class 1         0.88      0.88      0.88       504
accuracy                            0.93      1600
macro avg       0.91      0.91      0.91      1600
weighted avg    0.93      0.93      0.93      1600
Final Test Accuracy: 0.9263






Notes


Feature Mapping: The my_map function creates a 121-dimensional feature space, which may be computationally intensive for large datasets. The implementation is optimized using numpy for efficiency.
Weight Decoding: The my_decode function is partially implemented, with r and s vectors as placeholders. You may need to modify this function based on specific requirements for decoding the weights.
Hyperparameters: The logistic regression model uses C=0.01, solver='liblinear', and max_iter=100. Adjust these in the my_fit function if needed.
Error Handling: The code includes basic error handling for missing files and invalid input dimensions. Expand this as needed for robustness.
