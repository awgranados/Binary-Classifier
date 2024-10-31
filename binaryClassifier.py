import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

class Binary_Classifier(object):

    def __init__(self, train_data, train_target):
        self.train_data = np.array(train_data)
        self.train_target = np.array(train_target)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

    def logistic_training(self, alpha, lam, nepoch, epsilon):
        n, d = self.train_data.shape
        k = n // 5  
        best_params = None
        best_accuracy = 0

        for alpha_val in alpha:
            for lam_val in lam:
                self.weight = np.zeros(self.train_data.shape[1] + 1)

                accuracies = []
                for fold in range(5):
                    # Split data 
                    val_indices = list(range(fold * k, (fold + 1) * k))
                    train_indices = [i for i in range(n) if i not in val_indices]

                    X_train = self.train_data[train_indices]
                    y_train = self.train_target[train_indices]
                    X_val = self.train_data[val_indices]
                    y_val = self.train_target[val_indices]

                    # Feature scaling
                    X_train /= np.max(X_train, axis=0)
                    X_val /= np.max(X_val, axis=0)

                    train_data_aug = np.c_[X_train, np.ones(X_train.shape[0])]
                    val_data_aug = np.c_[X_val, np.ones(X_val.shape[0])]

                    prev_accuracy = 0.0
                    no_improvement_count = 0  

                    for epoch in range(nepoch):
                        shuffle_idx = np.random.permutation(len(X_train))
                        train_data_fold = train_data_aug[shuffle_idx]
                        train_target_fold = y_train[shuffle_idx]

                        for i in range(0, len(train_data_fold), 8):
                            batch_data = train_data_fold[i:i+8]
                            batch_target = train_target_fold[i:i+8]

                            gradient = np.dot(batch_data.T, ((self.sigmoid(np.dot(batch_data, self.weight))) - batch_target)) / len(batch_data)
                            normalization_factor = np.linalg.norm(self.weight[:-1])  # Compute normalization factor
                            regularization = lam_val * (self.weight[:-1] / (normalization_factor + 1e-8))  # Normalized L2 regularization
                            regularization_gradient = np.append(regularization, 0)
                            gradient += regularization_gradient

                            self.weight -= alpha_val * gradient

                        val_pred = self.sigmoid(np.dot(val_data_aug, self.weight))
                        val_pred = (val_pred > 0.5).astype(int)
                        accuracy = np.mean(val_pred == y_val)
                        accuracies.append(accuracy)

                        if accuracy - prev_accuracy < epsilon:
                            no_improvement_count += 1
                        else:
                            no_improvement_count = 0

                        if no_improvement_count >= 3:  
                            break

                        prev_accuracy = accuracy

                    avg_accuracy = np.mean(accuracies)

                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_params = (alpha_val, lam_val, self.weight)

        # Train with best parameters from entire dataset
        self.alpha, self.lam, self.w = best_params
        for epoch in range(nepoch):
            y_pred = self.sigmoid(np.dot(self.train_data, self.w[:-1]))
            grad = np.dot(self.train_data.T, y_pred - self.train_target) / n
            reg_term = self.lam * self.w[:-1]
            grad += reg_term
            self.w[:-1] -= self.alpha * grad

        print("Best Accuracy (Logistic Regression):", best_accuracy)
        return best_params

    def logistic_testing(self, testX):
        weights_with_bias = np.append(self.w[:-1], [self.w[-1]])
        testX_with_bias = np.c_[testX, np.ones(testX.shape[0])]
        y_pred = self.sigmoid(np.dot(testX_with_bias, weights_with_bias))
        return (y_pred >= 0.5).astype(int)

    def svm_training(self, gamma, C):
        train_data, val_data, train_target, val_target = train_test_split(self.train_data, self.train_target, test_size=0.1, random_state=0)

        best_accuracy = 0
        best_params = None

        for curr_gamma in gamma:
            for curr_C in C:
                # Init SVC with grid search parameters
                svc = SVC(kernel='rbf', gamma=curr_gamma, C=curr_C)

                # Train SVM
                svc.fit(train_data, train_target)

                # Eval accuracy on validation set
                accuracy = svc.score(val_data, val_target)

                # Check if current parameters are better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (curr_gamma, curr_C)
                    self.svm_model = svc

        # Store best parameters
        self.best_gamma, self.best_C = best_params
        print("Best Accuracy (SVM):", best_accuracy)

    def svm_testing(self, testX):
        pred_labels = self.svm_model.predict(testX)
        return pred_labels.reshape(-1, 1)

# Dataset preparation: Dataset is divided into 90% and 10%
# 90% for you to perform n-fold cross validation and 10% for autograder to validate your performance.
################## PLEASE DO NOT MODIFY ANYTHING! ##################
dataset = load_breast_cancer(as_frame=True)
train_data = dataset['data'].sample(frac=0.9, random_state=0)  # Random state is a seed value
train_target = dataset['target'].sample(frac=0.9, random_state=0)  # Random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)

# Model training: You are allowed to change the last two inputs for model.logistic_training
################## PLEASE DO NOT MODIFY ANYTHING ELSE! ##################
model = Binary_Classifier(train_data, train_target)

# Logistic Regression
logistic_start = time.time()
model.logistic_training([10**-10, 10], [10e-10, 1e10], 300, 0)
logistic_end = time.time()

# SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()

# Print training times
print(f"Training Time (Logistic Regression): {logistic_end - logistic_start:.4f} seconds")
print(f"Training Time (SVM): {svm_end - svm_start:.4f} seconds")
