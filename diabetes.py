import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabtes_dataset  = pd.read_csv('diabetes.csv')
X = diabtes_dataset.drop(columns = 'Outcome',axis=1)
Y = diabtes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify =Y,random_state =2)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train,Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)

import pickle

pickle.dump(classifier,open('model.pkl','wb'))

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))


# Input data
input_data = np.array([5,166,72,19,175,25.8,0.587,51])

# Reshape the input data to match the model's expected shape
input_data_reshaped = input_data.reshape(1, -1)

# Scale the input data using the same scaler used during training
standard_input_data = scaler.transform(input_data_reshaped)

# Make predictions
prediction = model.predict(standard_input_data)

print(prediction)