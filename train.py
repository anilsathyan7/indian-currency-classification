import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score 

# Configure file paths
MODEL=sys.argv[1]
DATA='model/data_600.npy'
LABEL='model/label_600.npy'

# Load data and labels
data=np.load(DATA)
label=np.load(LABEL)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

# Initialize a random forest classifier
print("\n<====Model====>\n")
clf = RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=1)
print(clf)

print("\n<====Training====>\n")

# Train the model
clf.fit(x_train, y_train)

# Check accuracy on test data
y_pred = clf.predict(x_test)
y_true = y_test
print("\nAccuracy:", accuracy_score(y_true,y_pred))

# Compute the confusion matrix
print("\n<====Confusion Matrix====>\n")
cf=confusion_matrix(y_true, y_pred)
print(cf)

print("\n<====Cross validation====>\n")

# Evaluate the model using five fold cross validation
scores = cross_val_score(clf, x_train, y_train, cv=5)  
print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Save the trained model
joblib.dump(clf, MODEL)

'''
Sample run: python train.py model/rfclassifier_600.sav
'''
