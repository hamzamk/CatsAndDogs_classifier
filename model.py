import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import pandas as pd
from kaggle_data_splitter import kaggle_data_splitter


def load_device()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print('Num of GPUs available: ', len(physical_devices))  

    try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Memory growth controlled') 
    except:
    # if GPU can't be used or CUDA error
    print('Device error')

def load_data(paths, dim, class_names, batch_size, augmentation=False):
    print('check paths: ', paths)
    if augmentation==False:
        datagen = ImageDataGenerator(rescale=1./255) #normalize images )
        valid_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
    if augmentation == True:
        datagen = ImageDataGenerator(
        rescale=1./255, #normalize images
        rotation_range = 30, #randomly rotate images in the range (0, 180)
        width_shift_range = 0.4, 
        height_shift_range = 0.3,
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.1,
        shear_range=0.1
    )
        valid_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
    train = datagen.flow_from_directory(paths[0], target_size = dim, color_mode='grayscale', classes = class_names, batch_size = batch_size, shuffle=True)
    valid = valid_datagen.flow_from_directory(paths[1], target_size = dim, color_mode='grayscale',  classes = class_names, batch_size = batch_size, shuffle=True)
    test = test_datagen.flow_from_directory(paths[2], target_size = dim,  color_mode='grayscale', classes = class_names, batch_size = batch_size, shuffle=False)
    # testset shuffle false as confusion matrix requires unshuffled labels
    return train, valid, test

    def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize= (20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def create_model():
    model = Sequential([
        Conv2D(filters = 16, kernel_size = (4, 4), activation = 'relu', padding = 'same', input_shape=(224, 224, 1)),
        MaxPool2D(pool_size = (5, 5), strides=2),
        Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (5, 5), strides=2),
        Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (5, 5), strides=2),
        Conv2D(filters = 128, kernel_size = (5, 5), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (5, 5), strides=2),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        Dense(units=64, activation='relu'),
        Dense(units=2, activation='softmax')
    ])
    return model

def create_callbacks(cp_path, tb_folder):
    path_ = os.getcwd()
    # Create a callback that saves the model's weights every 10 epochs
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=10)

    call_tensorboard = TensorBoard(log_dir=path_+'/logs/{}'.format(tb_folder))

    return [call_tensorboard, cp_callback]

def train_model():
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x=train_batches, validation_data=valid_batches, epochs=60, verbose=1, callbacks = create_callbacks(checkpoint_path, tb_name))

def test_model():
    names=['cats', 'dogs']
    from plot_confusion_matrix import plot_confusion_matrix
    test_imgs, test_labels = next(train_batches)
    print('test set loaded: ' , test_batches.classes.shape)
    plotImages(test_imgs)
    predictions = model.predict(x = test_batches, verbose=0)
    print('predictions acquired with tensor shape: ', predictions.shape)
    print('plotting confusion matrix and generating classification report')
    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=1)) 
    plot_confusion_matrix(cm, names, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues)
    print('Classification Report')
    report = classification_report(test_batches.classes, np.argmax(predictions, axis=1)
                                , target_names=names, output_dict=True)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.insert(0, 'Index', ['cats', 'dogs', 'accuracy', 'macro avg', 'weighted avg' ])
    df.to_csv('classification_report.csv', index=False)
    print('\n\n classification report saved to csv file')

def save_model(path):
    print('model saved in '{}'.format(path) directory')
    model.save(path)

def load_model(path):
    model = load_model(path)

def main():
    #Initialize GPU
    load_device()

    #Paths to image data
    DATADIR = os.path.join(os.getcwd(), 'dataset', 'kaggle')
    paths = [os.path.join(DATADIR + '\\train'), os.path.join(DATADIR+ '\\valid'), os.path.join(DATADIR+ '\\test')]

    #paths to save callbacks for tensorboard and checkpoints
    tb_name = 'fit'
    checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)           

    #load Data
    train_batches, valid_batches, test_batches = load_data(paths=paths, dim=(224, 224), 
                                                         class_names=['cats', 'dogs'], batch_size=32)

    #create model
    model = create_model()
    model.summary()

    #start training
    train_model()

    #test model and generate classification report and confusion matrix
    class_names=['cats', 'dogs']
    test_model(class_names)

    #save model
    save_model('./models')

    #test model using saved weights
    model = load_model('./models')
    test_model()
if __name__ == '__main__':
    main()
