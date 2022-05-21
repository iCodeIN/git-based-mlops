import os
import sys
import glob
import yaml
import random
import tarfile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.optimizers import Adam

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)

params = yaml.safe_load(open("params.yaml"))["train"]
print(params)

train = 'data'/Path(params['train'])
test = 'data'/Path(params['test'])
output = Path(sys.argv[2])
random.seed(params['seed'])

_image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def _get_serve_image_fn(model):
  @tf.function
  def serve_image_fn(image_tensor):
    return model(image_tensor)

  return serve_image_fn

def _get_signature(model): 
  signatures = {
      'serving_default':
          _get_serve_image_fn(model).get_concrete_function(
              tf.TensorSpec(
                  shape=[None, 224, 224, 3],
                  dtype=tf.float32,
                  name='image'))
  }

  return signatures  

def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, _image_feature_description)
    image = tf.io.decode_png(features['image'], channels=3) # tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.image.resize(image, [224, 224])
    image = resnet50.preprocess_input(image)

    label = tf.cast(features['label'], tf.int32)

    return image, label

def _read_dataset(epochs, batch_size, channel):
    filenames = glob.glob(str(channel/'*.tfrecord'))
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=4)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def _freeze_model_by_percentage(model, percentage):
  if percentage < 0 or percentage > 1:
    raise ValueError('Freeze percentage should between 0.0 and 1.0')

  if not model.trainable:
    raise ValueError(
        'The model is not trainable, please set model.trainable to True')

  num_layers = len(model.layers)
  num_layers_to_freeze = int(num_layers * percentage)

  for idx, layer in enumerate(model.layers):
    if idx < num_layers_to_freeze:
      layer.trainable = False
    else:
      layer.trainable = True

def _build_keras_model(num_class=10):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='max')

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(num_class, activation='softmax')
    ])

    return model, base_model

def _compile(model_to_fit, 
             model_to_freeze, 
             freeze_percentage,
             learning_rate):
  _freeze_model_by_percentage(model_to_freeze, freeze_percentage)

  model_to_fit.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=Adam(learning_rate=learning_rate),
      metrics=['sparse_categorical_accuracy'])

  return model_to_fit, model_to_freeze

def run_train():
    steps_per_epoch = int(params['train_size'] / params['batch_size'])
    total_epochs = int(params['train_num_steps'] / steps_per_epoch)

    if params['epoch'] > total_epochs:
        raise ValueError('Classifier epochs is greater than the total epochs')

    train_ds = _read_dataset(params['epoch'], params['batch_size'], train)
    # test_ds = _read_dataset(params['epoch'], params['batch_size'], test)

    model, base_model = _build_keras_model()
    model, base_model = _compile(model, 
                                 base_model, 
                                 1.0, 
                                 float(params['base_lr']))

    model.fit(
        train_ds,
        epochs=params['epoch'],
        steps_per_epoch=steps_per_epoch)

    model, base_model = _compile(model, 
                                 base_model, 
                                 params['finetune_freeze_pct'], 
                                 float(params['finetune_lr']))

    model.fit(
        train_ds,
        epochs=params['epoch'],
        steps_per_epoch=steps_per_epoch)

    model.save(output, 
               save_format='tf', 
               signatures=_get_signature(model)) 

run_train()    