
# Packages and Dictionaries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score



# Data Collection and Analysis
parkinsons_data = pd.read_csv('parkinsons.csv')

# print the first five rows of dataframe
parkinsons_data.head()

# Number of rows and columns in the dataform
parkinsons_data.shape

# Getting nore information about the dataset
parkinsons_data.info()

# Checking for missing values in each column
parkinsons_data.isnull().sum()

# Getting some statistical measures about the data
parkinsons_data.describe()

# Distribution of target Variable
parkinsons_data['status'].value_counts()

# grouping the data based on the target variable
parkinsons_data.groupby('status').mean()

# Separating the features adn Target
X = parkinsons_data.drop(columns=['name', 'status'], axis = 1)
Y = parkinsons_data['status']


print(X)

print(Y)

#Spliting the data to training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

# Data Standartization
scaler = StandardScaler()

scaler.fit(X_train)


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


print(X_train)

# Support Vector Machine Model
model = svm.SVC(kernel = 'linear')


# Training the SVM model with training data
model.fit(X_train, Y_train)


# Accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data: ', training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data: ', test_data_accuracy)


# Building a predictive System
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_in_np = np.asarray(input_data)

# Rehsape the np array
input_data_reshaped = input_data_in_np.reshape(1, -1)

# Standartize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
    print("The Person does not have Parkinson Disease")
else:
    print("The person has Parkinson")


