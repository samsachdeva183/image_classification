import sys
import os
import json
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import pandas as pd
from callbacks import callback_factory
from models import models_factory

batchsize = 16
trainsize = 1000
validationsize = 160


def setup_for_transfer_learning(base_model, model, lr):
    sgd = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0)
    model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

def setup_for_finetuning(base_model, model, lr):
    for layer in base_model.layers:
        layer.trainable = True

    sgd = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0)
    model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

def train(training_dir, validation_dir,lr,epochs,image_size,model_type,checkpoints_dir):

    model, base_model = models_factory(model_type,image_size)

    ## Optimizers
    '''
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    rmsprop = "rmsprop"
    sgd = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0)
    model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
    '''

    ## Callbacks
    checkpoint_name = os.path.join(checkpoints_dir,"weights_0.hdf5")
    history_path = os.path.join(checkpoints_dir, "history.csv")
    checkpointer = callback_factory('model_checkpoint')(checkpoint_name)
    early_stopping = callback_factory('early_stopping')(patience = 15)
    lr_scheduler = callback_factory('lr_kaggle')(lr)
    history_writer = callback_factory('history_writer')(history_path)

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=image_size,
            batch_size=batchsize,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=image_size,
            batch_size=batchsize,
            class_mode='binary')

    setup_for_transfer_learning(base_model, model, lr)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=trainsize // batchsize,
            epochs=min(5, epochs),
            validation_steps = validationsize // batchsize,
            validation_data=validation_generator,
            callbacks = [checkpointer, early_stopping, lr_scheduler, history_writer])
    
    setup_for_finetuning(base_model, model, lr)

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=trainsize // batchsize,
            epochs=max(0,epochs-5),
            validation_steps = validationsize // batchsize,
            validation_data=validation_generator,
            callbacks = [checkpointer, early_stopping, lr_scheduler, history_writer])

    print ("Training complete")


def main():

    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    training_dir = conf['training_dir']
    validation_dir = conf['validation_dir']
    lr = conf['learning_rate']
    epochs = conf['epochs']
    image_size = conf['image_size']
    model_type = conf['model_type']
    checkpoints_dir = conf['checkpoints_dir']
    

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    train(training_dir, validation_dir,lr,epochs,image_size,model_type,checkpoints_dir)


if __name__ == "__main__":
    main()