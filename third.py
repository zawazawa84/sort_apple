from keras.models import load_model
import pickle
import cv2
import glob

#モデルとクラス名の読み込み
model = load_model('cnn.h5')
classes = pickle.load(open('classes.sav', 'rb'))

#sample画像の前処理

img = cv2.imread('img/  stale_apple/rotated_by_15_Screen Shot 2018-06-07 at 2.24.47 PM.png')
img = cv2.resize(img,dsize=(224,224))
img = img.astype('float32')
img /= 255.0
img = img[None, ...]
result = model.predict(img)

#確率が一番大きいクラス
pred = result.argmax()

pred2 = str(classes[pred])
print(pred2)