
# Classification model to predict the gender (male or female) based on different acoustic parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Collect the data

from google.colab import drive
drive.mount('/content/drive')

voice=pd.read_csv('/content/drive/MyDrive/voice.csv')
voice.head()

# Pre-process the data

# Removing null values from the dataset

print(voice.isnull().sum())

# percentage distribution of label on a pie chart

# Count the number of male and female labels
label_counts = voice['label'].value_counts()

# Print the label counts
print(label_counts)

# Create labels and values for the pie chart
labels = ['male', 'female']
values = [1584,1584]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%')

# Add a title to the chart
plt.title('Label Distribution')

# Display the chart
plt.show()

##Split the Data

X = voice.drop('label', axis=1) 
Y = voice['label'] 

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Training & Evaluating the models

# a.Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

dt_pred = dt.predict(X_test)

print("Confusion matrix for Decision Tree:")
print(confusion_matrix(Y_test, dt_pred))

print("\nClassification report for Decision Tree:")
print(classification_report(Y_test, dt_pred))

# b.Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)

print("\nConfusion matrix for Random Forest:")
print(confusion_matrix(Y_test, rf_pred))

print("\nClassification report for Random Forest:")
print(classification_report(Y_test, rf_pred))

# c.KNN Classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)

print("\nConfusion matrix for KNN:")
print(confusion_matrix(Y_test, knn_pred))

print("\nClassification report for KNN:")
print(classification_report(Y_test, knn_pred))

# d.Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)

print("\nConfusion matrix for Logistic Regression:")
print(confusion_matrix(Y_test, lr_pred))

print("\nClassification report for Logistic Regression:")
print(classification_report(Y_test, lr_pred))

# e.SVM Classifier

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, Y_train)
svm_pred = svm.predict(X_test)

print("\nConfusion matrix for SVM:")
print(confusion_matrix(Y_test, svm_pred))

print("\nClassification report for SVM:")
print(classification_report(Y_test, svm_pred))

# Accuracy of the Models

from sklearn.metrics import accuracy_score

# Calculate the accuracy scores for each model
dt_accuracy = accuracy_score(Y_test, dt_pred)
rf_accuracy = accuracy_score(Y_test, rf_pred)
knn_accuracy = accuracy_score(Y_test, knn_pred)
lr_accuracy = accuracy_score(Y_test, lr_pred)
svm_accuracy = accuracy_score(Y_test, svm_pred)

# Print the accuracy scores
print("Accuracy score for Decision Tree      :", dt_accuracy)
print("Accuracy score for Random Forest      :", rf_accuracy)
print("Accuracy score for KNN                :", knn_accuracy)
print("Accuracy score for Logistic Regression:", lr_accuracy)
print("Accuracy score for SVM                :", svm_accuracy)

data =  {
    "Decision Tree"         : dt_accuracy,
    "Random Forest"         : rf_accuracy,
    "KNN"                   : knn_accuracy,
    "Logistic Regression"   : lr_accuracy,
    "SVM"                   : svm_accuracy,
    
}

courses = list(data.keys())
values  = list(data.values())
fig = plt.figure(figsize = (10, 5))

plt.bar(courses, values,color ='brown',width = 0.4)
plt.xlabel("Models")
plt.ylabel("Testing accuracy")
plt.title("Testing accuracy v/s Models")
plt.show()

# Determine the model with the best accuracy
best_model = max(dt_accuracy, rf_accuracy, knn_accuracy, lr_accuracy, svm_accuracy)

if best_model == dt_accuracy:
    print("\nThe best model is Decision Tree.")
elif best_model == rf_accuracy:
    print("\nThe best model is Random Forest.")
elif best_model == knn_accuracy:
    print("\nThe best model is KNN.")
elif best_model == lr_accuracy:
    print("\nThe best model is Logistic Regression.")
elif best_model == svm_accuracy:
    print("\nThe best model is SVM.")
