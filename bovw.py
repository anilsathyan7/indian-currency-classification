import numpy as np
import cv2
import mahotas
import os
import sys
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Setup filepath and parameters
IMDIR=sys.argv[1]
BOVW="model/bovw_codebook_600.pickle"
DICT_SIZE=600
DATA='model/data_600.npy'
LABEL='model/label_600.npy'

# Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def fd_haralick(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram
def fd_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins=8
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# Sift Bag of Visual Words
def feature_extract(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    feature = bowDiction.compute(gray, sift.detect(gray))
    return feature.squeeze()

# Directory containing images
base= Path(IMDIR).resolve()
print(base)

# Initialize SIFT and BOW trainer
sift = cv2.xfeatures2d.SIFT_create()
dictionarySize = DICT_SIZE
BOW = cv2.BOWKMeansTrainer(dictionarySize)

# Generate the vocabulary for BOVW
for file in base.glob('**/*.*'):
    fpath=Path(file).resolve()
    print(str(fpath))
    image = cv2.imread(str(fpath))
    print(fpath.parent.name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    kp, dsc= sift.detectAndCompute(gray, None)
    BOW.add(dsc)

# Cluster the visual vocabulary and generate codebook
print("Clustering...")
dictionary = BOW.cluster()
print((dictionary.shape))

# Save the codebook with pickle (for later use)
pickle_out = open(BOVW,"wb")
pickle.dump(dictionary, pickle_out)
pickle_out.close()


# Load the bovw codebook (verify)
pickle_in = open(BOVW,"rb")
dictionary = pickle.load(pickle_in)

# Initialize SIFT BOW image descriptor extractor
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

print("Feature Extraction...")

x_data=[]
x_label=[]
# Combine global features with SIFT-BOVW
for file in base.glob('**/*.*'):
    fpath=Path(file).resolve()
    image = cv2.imread(str(fpath))

    Humo=fd_hu_moments(image)
    Harl=fd_haralick(image)
    Hist=fd_histogram(image)
    Bovw=feature_extract(image)

    print(fpath.parent.name)
    mfeature= np.hstack([Humo, Harl, Hist, Bovw])
    print(mfeature.shape)
    x_data.append(mfeature)
    x_label.append(int(fpath.parent.name))

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
x_data = scaler.fit_transform(x_data)

# Encode labels
encoder  = LabelEncoder()
x_label  = encoder.fit_transform(x_label)

# Save the data and labels
np.save(DATA,np.array(x_data))
np.save(LABEL,np.array(x_label))

'''
Sample run: python bovw.py data
'''
