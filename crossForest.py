from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy as sp
import os
import os.path as path
import shutil
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import random
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Hyperparameter Tuning and Cross Validation Score amongst Random Forest
"""
# Data Preprocessing

main_path = './Data/random'

# Total Data: 30700 Files

# Random Total: 988 + 914

bodyParts = ['elbow', 'spine', 'humerus', 'legs', 'hip', 'forearm', 'hands', 'knee', 'shoulders']
sets = ['train', 'valid', 'test']
conditions = ['fractured', 'non-fractured']

# Preprocess
# Goes through every data so it can prepprocess them. It resizes them into 128x128 and converts them to grayscale, 
# It flattens the image and divides each value by 255, so the processor can read theme.
# it also labels them to see if they are fractured or not.

def preprocess(path):
    image_list = []
    labels = []
    for set in sets:
        for condition in conditions:
            folder_path = os.path.join(path,set, condition)
            # All the images
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                    
                image = Image.open(image_path)
                image = image.resize((128, 128))
                # Normalize Grayscape Images
                image = image.convert('L')  
            
                image_array = np.array(image).flatten()  
                    
                # 0 to 255 represents the color that is used in image processing.
                image_array = image_array / 255
                    
                image_list.append(image_array)
                if condition == 'fractured':
                    labels.append(1)
                else:
                    labels.append(0)
    
    X = np.array(image_list)
    y = np.array(labels)
    
    return X, y

X, y = preprocess(main_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define Hyperparameter Space
param_dist = {'n_estimators':[50, 100, 150], 'max_depth':[10, 20, 30], 'min_samples_split':[2, 5], 'min_samples_leaf':[1, 2]}

# Store best results
best_score = 0
best_params = None

results={
    'n_estimators': [],
    'max_depth': [],
    'min_samples_split': [],
    'min_samples_leaf':[],
    'train_score': [],
    'test_score': []
}

classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)

# Iteratively exhust various hyperparameter combinations
# Random Search CV

random_search = RandomizedSearchCV(estimator=classifier_rf, 
                                   param_distributions= param_dist,
                                   n_iter=10,  # Number of random combinations to sample
                                   cv=3,  # Number of folds for cross-validation
                                   verbose=2,  # Show progress
                                   n_jobs=-1,  # Use all available cores
                                   scoring='accuracy',
                                   random_state=42
)

random_search.fit(X_train, y_train)

print('Best Score: ', random_search.best_score_)
print('Best Params: ', random_search.best_params_)

best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)



