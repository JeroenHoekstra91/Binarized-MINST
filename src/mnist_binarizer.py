#!/usr/bin/python

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class MNIST_BINARIZER(object):

  # Required storage space (for dtype='bool'):
  # 70Kb in labels
  # 54.8Mb in images

  def convert(self):
    for split in ['train', 'test']:
      images_file_name = split + '-images.txt'
      labels_file_name = split + '-labels.txt'

      dataset = tfds.load('mnist', split=split, as_supervised=True)
      num_examples = len(dataset)

      b_images = np.ndarray((num_examples,) + (28, 28, 1), dtype="uint8")
      labels = np.ndarray(num_examples, dtype="uint8")

      for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
        b_images[i] = self.binarize(image)
        labels[i] = label

      with open(images_file_name, 'wb') as images_file, open(labels_file_name, 'wb') as labels_file:
        b_images.ravel().tofile(images_file, sep='')
        labels.tofile(labels_file, sep='')

  # Based on https://github.com/blei-lab/edward/blob/081ea532a982e6d2c88da25d6e2527f6a66f09ab/examples/vae.py#L38
  def binarize(self, image):
    normalized_image = image.astype(np.float32) / 255.0
    return np.random.binomial(1, normalized_image) # Draw a binary image using a Bernoulli model.
