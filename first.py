import os
import cv2
import numpy as np
import glob as glob
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

path = "img"
folders = os.listdir(path)

classes = [f for f in folders if os.path.isdir(os.path.join(path, f))]
n_classes = len(classes)

x=[]
y=[]

for label,class_name in enumerate(classes):
    files = glob.glob(path + "/" + class_name + "/*.png")
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, dsize=(224, 224))
        x.append(img)
        y.append(label)

x=np.array(x)
x=x.astype('float32')
x/=255.0

y=np.array(y)
y=np_utils.to_categorical(y, n_classes)
y[:5]

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)