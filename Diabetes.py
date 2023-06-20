import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
diabetes_dataset = pd.read_csv(r'D:\Team-Hextech\Mini-Prroject\Diabetes_Data\diabetes.csv')
diabetes_dataset.head()



X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2) 

classifier = svm.SVC(kernel = 'linear')

a=classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

pickle.dump(a,open('diabetes.pkl','wb'))

pickle.dump(scaler, open('scaler.pkl', 'wb'))
# while True:
#     print("Check whether you have diabetes or not")
#     a = []
#     a.append(eval(input("Enter the number of pregnancies:")))
#     a.append(eval(input("Enter your glucose level:")))
#     a.append(eval(input("Enter BP:")))
#     a.append(eval(input("Enter skin thickness:")))
#     a.append(eval(input("Enter insulin:")))
#     a.append(eval(input("Enter BMI:")))
#     a.append(eval(input("Enter Diabetes Pedigree Function:")))
#     a.append(eval(input("Enter age:")))

#     input_data = tuple(a)

#     # changing the input data into a numpy array
#     input_data_as_npa = np.asarray(input_data)

#     # reshape the array as we are predicting one instance
#     input_data_reshaped = input_data_as_npa.reshape(1, -1)

#     # we need to standardize the input data to match the trained data
#     std_data = scaler.transform(input_data_reshaped)
#     print(std_data)

#     prediction = classifier.predict(std_data)
#     print(prediction)

#     if prediction[0] == 0:
#         print('This person does not have diabetes')
#     else:
#         print('This person has diabetes')

#     choice = input("Do you want to continue? (y/n): ")
#     if choice.lower() != 'y':
#         break

# print("123")