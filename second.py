from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Activation, merging, Dense, Flatten, Dropout
from first import n_classes, x_train, x_test, y_train, y_test, classes
import pickle

input_tensor = Input(shape=(224,224,3))

base_model = VGG16(weights='imagenet', input_tensor=input_tensor,include_top=False)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(n_classes, activation='softmax'))

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

for layer in model.layers[:15]:
  layer.trainable = False

print('# layers=', len(model.layers))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=20, batch_size=16)
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)

#クラス名の保存
pickle.dump(classes, open('classes.sav', 'wb'))
#モデルの保存
model.save('cnn.h5')