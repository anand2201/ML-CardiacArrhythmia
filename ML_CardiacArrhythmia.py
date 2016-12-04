import pandas
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# var = input("Please enter something : ")
# print(var)


# Consuming the dataset using Pandas
data_df = pandas.read_csv('arrhythmia.data', header=None)

# Handling unknown values '?'. Replacing '?' with 'NaN'
data_df = data_df.replace('?', np.nan)
data_array = data_df.values

# Transposing the dataset to get the input vector 'X' by performing a transpose on the dataset
# After transpose all the rows except the last row are taken as the 'X' vector.
# Finally a transpose of the X vector is made so that the actual the input matrix is obtained
# The 'X' matrix is represented as 'NxM' matrix where 'N' represents no of rows and 'M' represents feature vectors
x = data_array.T[:-1, :].T

# The last row of the transpose of the dataset is taken as 'Y' and transposed. This is the label vector.
# The 'Y' matrix is represented as 'Nx1' matrix as it is the label for training for classification problem
y = data_array.T[-1, :].T

# The dataset contains unknown variables '?' which in the previous lines it is replaced with 'NaN'
# According to the Machine Learning paradigms the input vector training data shouldn't contain unknown variables
# as it is going to skew the results of the prediction models and certain models are not designed to perform prediction
# After research it is identified that the best possibility is to perform an activity called as imputation which is
# basically replacing the unknown variables with the mean of the feature vector (i.e. to take the mean of the column)
# and replacing the NaN with the mean value.
impute = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)

x_new = impute.fit_transform(x, y)
# Since the labels are obtained from a csv files the class labels are of type 'str'.Hence the labels are casted to 'int'
y_new = y.astype(int)

# Splitting the data into train and test set. This is a very naive and simple split of top 300 rows takens as train
# and the rest 100 is taken as the test set
# x_train = x_new[:352, :]
# y_train = y_new[:352]
# x_test = x_new[:-100, :]
# y_test = y_new[:-100]


# The raw input data 'X' and label 'Y' are given to a train_test_split function which basically splits the data into
# train and test vectors on the 'test_size' factor and the randomization factor for randomizing the dataset for split
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.10, random_state=50)

# Since this is a very high dimensional dataset there are many ways for the prediction to get skewed.
# 1. Noisy data
# 2. Unknown feature vector
# 3. Skewed feature vectors
# 4. High mean variance between feature groups which tends to skew the results
# Trying to perform dimensionality reduction for higher performance and achieve more accuracy
pca = PCA(n_components=23)

# Fitting the 'x_train' and 'x_test' data to be reduced to a lower component range
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# Perform Logistic regression inference on the 'x_train' and 'y_train'
# Logistic regression is performed using 'sag' solver and 'ovr -> one-vs-rest' or 'multinomial' classifier
# The objection function or the gradient descend is performed for a max iteration of 100
logistic_classifier = LogisticRegression(solver='sag', max_iter=1000, random_state=42, multi_class='multinomial').fit(
    x_train_pca, y_train)

# The final prediction matrix
y_predict = logistic_classifier.predict(x_test_pca)

print("Logistic Regression Classification Prediction : ")

# The training score of the Logistic regression estimator.
print("Training score for Logistic Regression Classifier: %.3f (%s)" % (
    logistic_classifier.score(x_train_pca, y_train), 'multinomial'))

# The accuracy of the Logistic regression classifier
print(accuracy_score(y_test, y_predict))

# Support Vector Machine Classifier is identified to be a better classifier for this classification problem
# Here I am trying to use a 'Linear' kernel for the converting the data for the non-linear separable data
# to a high dimensional space
svm_classifier = svm.SVC(kernel='linear')

# Training the SVM with the input 'x_train' and the label 'y_train' vectors for inference
svm_classifier.fit(x_train_pca, y_train)

# Performing a prediction of the 'x_test'
y_predict = svm_classifier.predict(x_test_pca)

print("SVM Classification Prediction : ")

# The training score of the SVM estimator to check how it performed while training or inference
print("Training score for SVM Classifier : %.3f (%s)" % (svm_classifier.score(x_train_pca, y_train), 'linear'))

# Final accuracy score of the svm estimator of the test data
print(accuracy_score(y_test, y_predict))

# ANOVA is a special type of filtering the input feature vector. The rationale behind this approach is to select the
# most appropriate feature vector to maximize the accuracy of the 'SVM' estimator
anova_filter = SelectKBest(f_regression, k=23)

# The kernel is specified to 'Linear' so that the data transformation by the SVM kernel is linear
anova_svm_classifier = svm.SVC(kernel='linear')

# 'make_pipeline' is pipeline function which executes a list of given function in sequential model. This is a process
# used to make the process of execution much more easier
anova_svm = make_pipeline(anova_filter, anova_svm_classifier)

# Training the 'SVM' estimator with the 'x_train' and 'y_train' to get the inference weight coefficients
anova_svm.fit(x_train, y_train)

# Final prediction of the trained 'SVM' estimator against the given 'x_test' vector
y_predict = anova_svm.predict(x_test)

print("ANOVA SVM Classification Prediction : ")

# Final accuracy score of the predicted data and the actual data
print(accuracy_score(y_test, y_predict))
