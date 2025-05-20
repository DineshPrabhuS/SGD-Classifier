# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries like sklearn, pandas, and matplotlib.

2.Load the Iris dataset and convert it into a DataFrame.

3.Split the data into features (X) and target (Y).

4.Divide data into training and testing sets.

5.Train the model using SGDClassifier on the training data.

6.Predict and evaluate using accuracy score and confusion matrix.

## Program:
/*
Program to implement the prediction of iris species using SGD Classifier.

Developed by: DINESH PRABHU S

RegisterNumber: 212224040077
*/
```
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the dataset
iris = load_iris()

# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)

y_pred = sgd_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)
```

## Output:
![image](https://github.com/user-attachments/assets/62ef29c0-f190-4116-9dbf-d07419f7b237)

![image](https://github.com/user-attachments/assets/90a5611a-e4ba-441c-ab7f-51988c962bf2)

![image](https://github.com/user-attachments/assets/71a98cdf-11e5-48b0-ad5a-89f23052a435)

![image](https://github.com/user-attachments/assets/0bac2ab8-0d6d-487d-9271-eddc7b9f4331)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
