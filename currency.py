import numpy as np
import cv2, mahotas, imutils
import os, sys
import math, pickle
from sklearn.externals import joblib

# Configure file paths
IMAGE=sys.argv[1]
MODEL='model/rfclassifier_600.sav'
BOVW="model/bovw_codebook_600.pickle"

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

# SIFT Bag of Visual Words
def feature_extract(im):
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    feature = bowDiction.compute(gray, sift.detect(gray))
    return feature.squeeze()


def rotate_bound(image,point,angle):

    '''
    Perform rotation on src image to make 'contour' horizontal (i.e it's longest side)
    The overall size of the rotated image increases to prevent border cutoff
    Returns the new 'contour-center' and the rotated image
    '''

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix and find their sine, cosine components
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    # find new position of the point after rotation
    point=np.array(point).reshape((1,1,-1))
    newpoint=cv2.transform(point, M).squeeze()

    # perform the actual rotation and return the image along with transformed point
    return newpoint,cv2.warpAffine(image, M, (nW, nH))


def crop_image(rect, src):

    # Get center, size, and angle from rect
    center, size, theta = rect
    width, height = int(size[0]), int(size[1])   

    # Make sure that largest side is the width
    if size[0] < size[1]: 
      theta += 90
      width,height=height,width
    
    print("Center: ",center, "Shape:",(width,height), "Theta:",theta)
    
    # Get the rotated image and new contour center
    newcenter, dst = rotate_bound(src, center, theta)

    # Crop the horizontal patch from rotated image
    out = cv2.getRectSubPix(dst, (width-ks,height-ks), tuple(newcenter))

    return out

# Load the trained model and input image
loaded_model = joblib.load(MODEL)

# Class-label dictionary and colours
label= {0:"10", 1:"20", 2:"50", 3:"100", 4:"200", 5:"500" , 6:"2000"}
colours=[(0, 63, 123),(47, 255, 173),(238, 244, 21),(220, 126, 181),(29, 170, 255),(133, 142, 146),(255,0, 255)]

# Load the BOVW codebook
pickle_in = open(BOVW,"rb")
dictionary = pickle.load(pickle_in)

# Initialize SIFT BOW image descriptor extractor
sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

def features(image):
  Humo=fd_hu_moments(image)
  Harl=fd_haralick(image)
  Hist=fd_histogram(image)
  Bovw=feature_extract(image)

  mfeature= np.hstack([Humo, Harl, Hist, Bovw])
  return(mfeature)


# Read input image and convert to grayscale
img = cv2.imread(IMAGE)
print("Image shape: " + str(img.shape)) 
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,7,7)
cv2.imwrite('binary_edge.png',thresh)

# Dilate the image
ks=int(math.log(img.shape[0]*img.shape[1],7))
print("Kernel size:",ks)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ks,ks))
thresh = cv2.dilate(thresh, kernel)
cv2.imwrite('dilated_edge.png',thresh)

# Find all contours of binary image
_ , contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Number of contours
num_contours=len(contours)
print("Total number of contours: " + str(num_contours))


# Crop and save proper contours separately [filter by area]
colour_crop=[]
mfeatures=[]
num_contours=0
img_size=320
min_area=img.shape[0]*img.shape[1]*0.05

for cnt in contours:
  if(cv2.contourArea(cnt)>min_area):
    
    # find min-bound rectangle
    rect = cv2.minAreaRect(cnt)
        
    # crop rotated contour
    crop = crop_image(rect, img)
    colour_crop.append(crop)
    cv2.imwrite("contour_crop_o"+str(num_contours)+".png",crop)
    num_contours=num_contours+1
    
    # Resize contour crop
    (height, width,channel) = crop.shape
    resize_ratio = 1.0 * (img_size / max(width, height))
    target_size = (int(resize_ratio * width), int(resize_ratio * height))

    curr = cv2.resize(crop, target_size)
   
    mfeatures.append(features(curr))
    
# Number of proper contours
print("Number of proper contours: " + str(num_contours))

# Predict contour labels
cluster=loaded_model.predict(mfeatures)
probs=loaded_model.predict_proba(mfeatures)
print("\nPredicted labels:- ", cluster)
print("Predicted probabilities:-\n", probs)


# Draw contours with labels
num_cnt=0
total=0
for cnt in contours:
  if(cv2.contourArea(cnt)>min_area):
    
    # Find minimum boundong rectangle
    rect = cv2.minAreaRect(cnt)
    # Center,width and height
    (x,y),(w,h)=rect[0],rect[1]
    
    # Draw bounding rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,colours[cluster[num_cnt]],2)

    # Set label colour and position
    center=int(x),int(y)
    cv2.putText(img,label[cluster[num_cnt]], center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 5)
    cv2.putText(img,label[cluster[num_cnt]], center, cv2.FONT_HERSHEY_SIMPLEX, 1, colours[cluster[num_cnt]], 3)
    
    # Compute total sum
    total=total+int(label[cluster[num_cnt]]) 
    num_cnt=num_cnt+1         

# Print total amount
print("Total sum: ", total)

# Show output in window
cv2.imshow("Bounding Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Sample run: python currency.py test/currency_test.jpg
'''
