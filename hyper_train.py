import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 

# Configure file paths
DATA='model/data_600.npy'
LABEL='model/label_600.npy'

# Load data and labels
data=np.load(DATA)
label=np.load(LABEL)

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)

# Initialize a random forest classifier
print("\n<====Model====>\n")
clf = RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=1)
print(clf)

print("\n<====Grid Search====>\n")

# Enumerate search parameters
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30, 50, 70, 100, 150]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

# Set up parameter dictionary
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

# Perform grid search
gridF = GridSearchCV(clf, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(x_train, y_train)

print("\n<====Best Model====>\n")

# Print the best parameters and score
print(bestF.best_score_)
print(bestF.best_params_)

'''
Sample run: python hyper_train.py
'''
