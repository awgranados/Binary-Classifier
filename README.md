# Project Documentation

## Running the Code

To run this binary classification project, ensure that Python is installed on your local machine. Install the necessary Python packages using pip with the following command:

```bash
pip install numpy scikit-learn matplotlib
```

Once all packages are installed, download the `binaryClassifier.py` file containing the implementation. Finally, execute the script using the command:

```bash
python binaryClassifier.py
```

## Code Overview

The code implements a binary classifier using logistic regression and support vector machine (SVM) models. It leverages the `scikit-learn` library for SVM training and testing, while a custom implementation is used for logistic regression. The logistic regression model employs mini-batch gradient descent with L2 regularization and early stopping based on validation accuracy.

### Model Training

The logistic regression model is trained using the following key hyperparameters:
- **Learning Rate (`alpha`)**: Controls the step size during optimization.
- **Regularization Parameter (`lam`)**: Helps prevent overfitting by penalizing large weights.
- **Number of Epochs (`nepoch`)**: Determines how many iterations the training runs.
- **Early Stopping Criteria (`epsilon`)**: Stops training if validation accuracy does not improve over a set number of epochs.

### Hyperparameter Tuning

For both models, a grid search approach is utilized to find the best hyperparameters:
- **Logistic Regression**: Trains with multiple combinations of `alpha` and `lam` values to identify the best performing model.
- **SVM**: Uses the radial basis function (RBF) kernel with varying `gamma` and `C` parameters.

### Training Time and Performance Metrics

The code also tracks and prints the training time for logistic regression and SVM, alongside the accuracy of the logistic regression model. This is useful for evaluating the efficiency and effectiveness of each model.
