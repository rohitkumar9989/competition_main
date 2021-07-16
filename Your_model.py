import tensorflow as tf
import numpy as np
import pandas as pd
class your_model ():
  def __init__ (self, filename, save_checkpoints=True, checkpoints_already_saved=False):
    self.filename=filename #Your filename whaere you images are present
    self.save_checkpoints=save_checkpoints
  def models_api (self):
    data_api=tf.keras.preprocessing.image_dataset_from_directory(directory=self.filename,
                                                             label_mode='binary',
                                                             batch_size=32,
                                                             image_size=(224, 224),
                                                             shuffle=False,
                                                             interpolation='bilinear',
                                                             smart_resize=True)

    data_augmented=tf.keras.Sequential([
                                        tf.keras.layers.experimental.preprocessing.RandomHeight(factor=0.2),
                                        tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
                                        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    model_base=tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    inputs=tf.keras.layers.Input(shape=(224, 224, 3), name='first_layer')
    augs=data_augmented(inputs)
    model=model_base(augs, training=False)

    layer_1=tf.keras.layers.GlobalMaxPool2D()(model)

    outputs=tf.keras.layers.Dense(128, activation='relu')(layer_1)
    outputs=tf.keras.layers.Dense(16, activation='relu')(outputs)
    outputs=tf.keras.layers.Dense(16, activation='relu')(outputs)

    output_real=tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
    model=tf.keras.Model(inputs, outputs)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics='accuracy')
    if self.save_checkpoints==True:
      model.fit(data_api, epochs=100, callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200)),
                                                 tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints.ckpt',
                                                                                    save_weights_onlt=True,
                                                                                    monitor='accuracy'),
                                                 tf.keras.callbacks.EarlyStopping(patience=10, verbose=1,
                                                                                  restore_best_weights=True,
                                                                                  monitor='accuracy')])
    else:
      model.fit(data_api, epochs=1)
      model.load_weights('checkpoints.ckpt')

    return model
  def model_cnn (self):
    data=tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                          featurewise_std_normalization=True,
                                                          zca_epsilon=1e-4,
                                                          shear_range=0.2,
                                                          height_shift_range=0.2,
                                                          width_shift_range=0.2,
                                                          fill_mode='nearest',
                                                          horizontal_flip=True)
    data=data.flow_from_directory(directory=self.filename,
                              target_size=(224, 224), 
                              class_mode='binary',
                              shuffle=False,
                              batch_size=32,
                              interpolation='bilinear'
                              )
    
    model_base=tf.keras.Sequential([
                         tf.keras.layers.Conv2D(filters=128,
                                                kernel_size=2,
                                                activation='relu',
                                                input_shape=(224, 224, 3)),
                         tf.keras.layers.Conv2D(128, 3, activation='relu'),
                         tf.keras.layers.Conv2D(128, 2, activation='relu'), 

                         tf.keras.layers.Conv2D(64, 3, activation='relu'),
                         tf.keras.layers.Conv2D(64, 3, activation='relu'),
                         tf.keras.layers.Conv2D(64, 3, activation='relu'),  

                         tf.keras.layers.Conv2D(32, 2, activation='relu'),
                         tf.keras.layers.MaxPool2D(pool_size=2,
                                                   padding='same',
                                                   data_format=None),
                         tf.keras.layers.Conv2D(32, 2, activation='relu'),

                         tf.keras.layers.Flatten(), 

                         tf.keras.layers.Dense(128, activation='relu'),
                         tf.keras.layers.Dense(128, activation='relu'),
                         tf.keras.layers.Dense(128, activation='relu'), 

                         tf.keras.layers.Dense(64, activation='relu'),
                         tf.keras.layers.Dense(64, activation='relu'),
                         tf.keras.layers.Dense(64, activation='relu'), 

                         tf.keras.layers.Dense(32, activation='relu'),
                         tf.keras.layers.Dense(32, activation='relu'),
                         tf.keras.layers.Dense(32, activation='relu'),  

                         tf.keras.layers.Dense(16, activation='relu'),
                         tf.keras.layers.Dense(16, activation='relu'),
                         tf.keras.layers.Dense(16, activation='relu'), 

                         tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_base.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00012),
                    metrics='accuracy')
    
    model_base.fit(data, 
                   epochs=20, 
                   steps_per_epoch=len(data),
                   callbacks=tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                              patience=10,
                                                              verbose=1))

    return model_base



    
    
#This is for the cnn model
a=your_model(filename='C:\\Users\\rohit\\Downloads\\Face_data')

#For cnn
#model=a.model_cnn()

#For api
#model=a.models_api()


model.save('Mymodel', save_format='h5')