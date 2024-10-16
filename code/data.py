import os
import numpy as np 
import pandas as pd 
import imageio
import tensorflow as tf



import matplotlib.pyplot as plt

def load(image_path,mask_path):
    '''you have to include the path to images not include the  end point '''
    image_list_orig = os.listdir(image_path)
    mask_list_orig =os.listdir(mask_path)
    image_list = [os.path.join(image_path,i) for i in image_list_orig]
    mask_list = [os.path.join(mask_path,i) for i in mask_list_orig]
    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    return image_list,mask_list,dataset



def show(image_id,image_list,mask_list):
    img = imageio.imread(image_list[image_id])
    mask = imageio.imread(mask_list[image_id])
    #mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])
    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :, 0])
    arr[1].set_title('Segmentation')



def input_data(dataset,BUFFER_SIZE=500,BATCH_SIZE=32):
    #sub_function for normalizing our data 
    def process_path(image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img, mask
    #sub_function for resize (we can use the size as argument in function)
    def preprocess(image, mask,image_size=(96, 128)):
        input_image = tf.image.resize(image, image_size, method='nearest')
        input_mask = tf.image.resize(mask, image_size, method='nearest')
        return input_image, input_mask
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset
