import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from plot_model_history import plot_model_history 
from confusion_matrix import plot_confusion_matrix
import pre_processing as pp
import matplotlib.pyplot as plt

num_epochs = 1
batch_size = 64

# CNN yapısı oluşturulur.
model = Sequential()

# Model 1
model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(pp.width, pp.height, 1), data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Model 2
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Model 3
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flatten
model.add(Flatten())

# Dense 1
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Dense 2
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Dense 3
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Output katmanı
model.add(Dense(pp.num_classes, activation='softmax'))

# Adam optimizerinin ayaarlanması.
model.compile(loss='categorical_crossentropy', optimizer=Adam(decay=1e-6), metrics=['accuracy'])

# Modelin ekranda gösterilmesi.
# model.summary()

# Verilerin çekilmesi.
train_X = np.load('../data/train_X.npy')
train_Y = np.load('../data/train_Y.npy')
test_X = np.load('../data/test_X.npy')
test_Y = np.load('../data/test_Y.npy')

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


earlyStop = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)

model_history = model.fit(data_generator.flow(train_X, train_Y, batch_size),
                                steps_per_epoch=len(train_X) / batch_size,
                                epochs=num_epochs,
                                verbose=2, 
                                callbacks = [earlyStop],
                                validation_data=(test_X, test_Y)
                                )

plot_model_history(model_history)
model.save_weights('../data/model.h5')

# Test performansının değerlendirilmesi.
test_true = np.argmax(test_Y, axis=1)
test_prediction = np.argmax(model.predict(test_X), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_prediction)))

# Karmaşıklık matrisinin elde edilmesi.
plot_confusion_matrix(test_true, test_prediction, classes=pp.emotion_counts.emotion, normalize=True, title='Normalized confusion matrix')