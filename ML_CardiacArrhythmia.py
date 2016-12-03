import pandas
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA

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
# According to the Machine Learning paradigms the input vector training data shouldnt contain unknown variables
# as it is going to skew the results of the prediction models and certain models are not designed to perform prediction
# After research it is identified that the best possibility is to perform an activity called as imputation which is
# basically replacing the unknown variables with the mean of the feature vector (i.e. to take the mean of the column)
# and replacing the NaN with the mean value.
impute = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
x_new = impute.fit_transform(x, y)
y_new = y.astype(int)

# x_train = x_new[:300, :]
# y_train = y_new[:300]
# x_test = x_new[:-100, :]
# y_test = y_new[:-100]

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.10, random_state=40)
pca = PCA(n_components=23)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

clf = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class='ovr').fit(x_train, y_train)
print("Logistic Regression Prediction :")
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print("training score : %.3f (%s)" % (clf.score(x_train, y_train), 'multinomial'))


lin_clf = svm.SVC(kernel='linear')
lin_clf.fit(x_train, y_train)
y_pred = lin_clf.predict(x_test)
# print(y_test)
# print(y_pred)
print("SVM Classification Prediction : ")
print(accuracy_score(y_test, y_pred))
print("training score : %.3f (%s)" % (lin_clf.score(x_train, y_train), 'linear'))
