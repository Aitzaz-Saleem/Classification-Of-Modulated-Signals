# -*- coding: utf-8 -*-


import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Import dataset
dataset = pd.read_excel('D:\Extra\Modulation_Classification\Features.xlsx' , sheet_name='1')


# # Handling missing values
# imputer = SimpleImputer(strategy='mean')
# dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# Handling missing values
dataset.dropna(inplace=True)


# Separating the features and labels 
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Label_Encoder object 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=0)

# Feature scaling 
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

# Fitting the model 
classifier = RandomForestClassifier(n_estimators=10, random_state=1)
classifier.fit(X_train,y_train)


y_pred1 = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred1)*100
testscore = classifier.score(X_test, y_test)
trainscore = classifier.score(X_train, y_train)

print("Accuracy Score is : {:.2f}%".format(acc))
print("Training Score is : {:.2f}".format(trainscore))
print("Testing Score is : {:.2f}".format(testscore))



# Classification Report for Classifier
print(f'Classification Report:\n{classification_report(y_test, y_pred1)}')

# Confusion Matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred1, normalize=False, figsize=(16,16), cmap='Blues')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)
plt.yticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()