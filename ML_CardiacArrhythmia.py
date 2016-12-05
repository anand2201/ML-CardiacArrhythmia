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


# Compute the PCA for the given set of components and return a list of x_train and x_test list of dimensional reduced
# feature vectors
def pca_computation(list_of_components, x_train_val, x_test_val):
    list_of_x_train = []
    list_of_x_test = []
    for val in list_of_components:
        current_pca = PCA(n_components=val)
        current_pca.fit(x_train_val)
        list_of_x_train.append(current_pca.transform(x_train_val))
        list_of_x_test.append(current_pca.transform(x_test_val))
    return list_of_x_train, list_of_x_test


input_list = ['1', '2', '3', '4', '5', '6']

print("Please type value from 1 to 5 to perform the below indicated operation: ")
print("1. Logistic Regression only")
print("2. Linear SVM only")
print("3. ANOVA SVM only")
print("4. All estimators")
print("5. Logistic Regression accuracy graph. This runs for different PCA dimensions. (This is time consuming)")
print("6. SVM accuracy graph. This runs for different PCA dimensions. (This is time consuming)")
var = input()

if var not in input_list:
    print("Please enter an integer from 1 to 5\n\n")
    sys.exit(2)

print("***************************Performing the Prediction of the dataset*********************")

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
# Note: I have enquired with professor to confirm that imputation is good data pre-processing strategy to handle unknown
# variables rather than removing the feature column or filling with 0. Imputation is taking the mean of the column and
# replacing the unknown variable '?' with the mean of the column
impute = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)

x_new = impute.fit_transform(x, y)
# Since the labels are obtained from a csv files the class labels are of type 'str'.Hence the labels are casted to 'int'
y_new = y.astype(int)

# Splitting the data into train and test set. This is a very naive and simple split of top 300 rows takens as train
# and the rest 100 is taken as the test set. This set is basically a sanity test split and it is not a good approach
# for prediction
# x_train = x_new[:352, :]
# y_train = y_new[:352]
# x_test = x_new[:-100, :]
# y_test = y_new[:-100]


# The raw input data 'X' and label 'Y' are given to a train_test_split function which basically splits the data into
# train and test vectors on the 'test_size' factor and the randomization factor for randomizing the dataset for split
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.10, random_state=100)

# Since this is a very high dimensional dataset there are many ways for the prediction to get skewed.
# 1. Noisy data
# 2. Unknown feature vector
# 3. Skewed feature vectors
# 4. High mean variance between feature groups which tends to skew the results
# Trying to perform dimensionality reduction for higher performance and achieve more accuracy
pca = PCA(n_components=80)

# Fitting the 'x_train' and 'x_test' data to be reduced to a lower component range
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
pca_components = [50, 60, 70, 80, 90]

if var in ['1', '4', '5']:

    # Perform Logistic regression inference on the 'x_train' and 'y_train'
    # Logistic regression is performed using 'sag' solver and 'ovr -> one-vs-rest' or 'multinomial' classifier
    # The objection function or the gradient descend is performed for a max iteration of 100
    logistic_classifier = LogisticRegression(solver='sag', max_iter=4000, random_state=42,
                                             multi_class='multinomial').fit(x_train_pca, y_train)

    # The final prediction matrix
    y_predict = logistic_classifier.predict(x_test_pca)

    print("****************************************************************************************\n")

    print("Logistic Regression Classification : ")

    # The training score of the Logistic regression estimator.
    print("Training score for Logistic Regression Classifier: %.3f (%s)" % (
        logistic_classifier.score(x_train_pca, y_train), 'multinomial'))

    # The accuracy of the Logistic regression classifier
    print("Accuracy score for Logistic Regression Classifier: %.3f \n" % (accuracy_score(y_test, y_predict) * 100))

    print("****************************************************************************************\n")

    if var == '5':
        print("Generating the PCA Components vs Accuracy Graph for Logistic Regression:\n")
        # Get the list of 'x_train' and 'x_test' for various PCA components declared at the script initialization
        x_train_pca_list, x_test_pca_list = pca_computation(pca_components, x_train, x_test)
        logistic_regression_accuracy_list = []
        for idx, value in enumerate(pca_components):
            x_train_pca_current = x_train_pca_list[idx]
            x_test_pca_current = x_test_pca_list[idx]
            # Performing the Logistic regression inference of the 'current_x_train' and 'y_train'
            logistic_classifier_current = LogisticRegression(solver='sag', max_iter=4000, random_state=10,
                                                             multi_class='multinomial').fit(x_train_pca_current,
                                                                                            y_train)
            # Final accuracy score of the current estimator.
            final_accuracy = accuracy_score(logistic_classifier_current.predict(x_test_pca_current), y_test) * 100
            logistic_regression_accuracy_list.append(final_accuracy)
        plt.plot(pca_components, logistic_regression_accuracy_list)
        plt.xlabel("PCA Components Range")
        plt.ylabel("Logistic Regression Accuracy Value")
        plt.show()
        print("****************************************************************************************\n")

if var in ['2', '4', '6']:
    # Support Vector Machine Classifier is identified to be a better classifier for this classification problem
    # Here I am trying to use a 'Linear' kernel for the converting the data for the non-linear separable data
    # to a high dimensional space
    svm_classifier = svm.SVC(kernel='linear')

    # Training the SVM with the input 'x_train' and the label 'y_train' vectors for inference
    svm_classifier.fit(x_train_pca, y_train)

    # Performing a prediction of the 'x_test'
    y_predict = svm_classifier.predict(x_test_pca)

    print("SVM Classification : ")

    # The training score of the SVM estimator to check how it performed while training or inference
    print("Training score for SVM Classifier : %.3f (%s)" % (svm_classifier.score(x_train_pca, y_train), 'linear'))

    # Final accuracy score of the svm estimator of the test data
    print("Accuracy score for SVM Classifier: %.3f \n" % (accuracy_score(y_test, y_predict) * 100))

    print("****************************************************************************************\n")

    if var == '6':
        print("Generating the PCA Components vs Accuracy Graph for SVM Classifier:\n")
        # Obtaining the 'x_train' and 'x_test' for various PCA components
        x_train_pca_list, x_test_pca_list = pca_computation(pca_components, x_train, x_test)
        svm_accuracy_list = []
        for idx, value in enumerate(pca_components):
            x_train_pca_current = x_train_pca_list[idx]
            x_test_pca_current = x_test_pca_list[idx]
            svm_classifier_current = svm.SVC(kernel='linear')
            # Performing the inference of the 'current_x_train' and 'y_train'
            svm_classifier_current.fit(x_train_pca_current, y_train)
            # Final accuracy of the current estimator
            final_accuracy = accuracy_score(svm_classifier_current.predict(x_test_pca_current), y_test) * 100
            svm_accuracy_list.append(final_accuracy)
        plt.plot(pca_components, svm_accuracy_list)
        plt.xlabel("PCA Components Range")
        plt.ylabel("SVM Accuracy Value")
        plt.show()
        print("****************************************************************************************\n")

if var in ['3', '4']:
    # ANOVA is a special type of filtering the input feature vector. The rationale behind this approach is to select the
    # most appropriate feature vector to maximize the accuracy of the 'SVM' estimator
    anova_filter = SelectKBest(f_regression, k=90)

    # The kernel is specified to 'Linear' so that the data transformation by the SVM kernel is linear
    anova_svm_classifier = svm.SVC(kernel='linear')

    # 'make_pipeline' is pipeline function which executes a list of given function in sequential model.
    # This is a process used to make the process of execution much more easier
    anova_svm = make_pipeline(anova_filter, anova_svm_classifier)

    # Training the 'SVM' estimator  with the 'x_train' and 'y_train' to get the inference weight coefficients
    anova_svm.fit(x_train, y_train)

    # Final prediction of the trained 'SVM' estimator against the given 'x_test' vector
    y_predict = anova_svm.predict(x_test)

    print("ANOVA SVM Classification : ")

    # Final accuracy score of the predicted data and the actual data
    print("Accuracy score for ANOVA SVM Classifier: %.3f \n" % (accuracy_score(y_test, y_predict) * 100))

    print("****************************************************************************************")
