#!/usr/bin/env python
# coding: utf-8

# In[7]:


import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Activation,Dropout
from keras.utils import normalize, to_categorical
from keras import backend as K
import numpy as np
from keras.preprocessing import image



img_width ,img_height=100,100

train_data_dir=r'C:\Users\Divyansh\Desktop\apple\data\train'
validation_data_dir=r'C:\Users\Divyansh\Desktop\apple\data\train'
batch_size=50

#Data augmentation

if K.image_data_format()=="channels_first":
	input_shape=(3,img_width,img_height)
else:
	input_shape=(img_width,img_height,3)


datagen= ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	validation_split=0.2)

train_generator=datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	subset="training",
	class_mode='categorical')

val_datagen =ImageDataGenerator(rescale=1./155)

validation_generator =datagen.flow_from_directory(train_data_dir,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	subset="validation",
	class_mode='categorical')

model = Sequential()

model.add(Conv2D(132,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))

model.add(Activation('softmax'))


	


# In[8]:


rms=keras.optimizers.RMSprop(learning_rate=0.05,rho=0.9)

model.compile(loss='categorical_crossentropy',
             optimizer=rms,
             metrics=['categorical_accuracy'])


# In[ ]:





# In[9]:


from keras.utils import plot_model
plot_model(model,to_file='model.png')


# In[24]:


from keras.callbacks import History

history=History()

model.fit(
    train_generator,
    steps_per_epoch=71,
    epochs=10,
    callbacks=[history],
    validation_data=validation_generator,
    validation_steps=12)


# In[25]:


print(history.history.keys())


# In[22]:


import matplotlib.pyplot as plt


plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[23]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[ ]:




