#!/usr/bin/python

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2

class MNIST_BINARIZER(object):

  # Required storage space (for dtype='bool'):
  # 70Kb in labels
  # 54.8Mb in images

  def convert(self, downscale=False):
    for split in ['train', 'test']:
      images_file_name = split + '-images.txt'
      labels_file_name = split + '-labels.txt'

      dataset = tfds.load('mnist', split=split, as_supervised=True)
      num_examples = len(dataset)
      image_shape = (14, 14, 1) if downscale else (28, 28, 1)

      b_images = np.ndarray((num_examples,) + image_shape, dtype="uint8")
      labels = np.ndarray(num_examples, dtype="uint8")

      for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
        image = self.binarize(image)
        if downscale: image = self.downscale(image)

        b_images[i] = image
        labels[i] = label

      with open(images_file_name, 'wb') as images_file, open(labels_file_name, 'wb') as labels_file:
        b_images.ravel().tofile(images_file, sep='')
        labels.tofile(labels_file, sep='')

  def downscale(self, image):
    return cv2.resize(image, dsize=(14, 14), interpolation=cv2.INTER_NEAREST).reshape(14, 14, 1)

  # Based on https://github.com/blei-lab/edward/blob/081ea532a982e6d2c88da25d6e2527f6a66f09ab/examples/vae.py#L38
  def binarize(self, image):
    normalized_image = image.astype(np.float32) / 255.0
    return np.random.binomial(1, normalized_image) # Draw a binary image using a Bernoulli model.
