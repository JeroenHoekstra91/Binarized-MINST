# MNIST Binarizer

This is a side project which creates a binarized version of the MNIST dataset

## System Dependencies

- [Python](https://www.python.org/) 3.6.9
- [pip](http://www.pip-installer.org)
- virtualenv (`$ pip install virtualenv` OPTIONAL)

## Installation

```console
$ git clone https://github.com/JeroenHoekstra91/Binarized-MINST.git
$ cd Binarized-MINST
$ virtualenv venv --python=python3.6.9 (OPTIONAL)
$ source venv/bin/activate (OPTIONAL)
$ pip install -r requirements.txt
```

## Usage 

For converting the original MNIST dataset into a binarized version (including labels):

```python
from src.mnist_binarizer import MNIST_BINARIZER
converter = MNIST_BINARIZER()
converter.convert()
```

For reading the images, assuming they are in the same directory as the working directory:

```python
import numpy as np

# For the training set
num_examples = 60000

with open('train-labels.txt', 'rb') as f:
  train_labels = np.fromfile(f, dtype='uint8', count=-1, sep='')

with open('train-images.txt', 'rb') as f:
  train_images = np.fromfile(f, dtype='bool', count=-1, sep='').reshape((num_examples,) + (28, 28, 1))

# For the test set
num_examples = 10000

with open('test-labels.txt', 'rb') as f:
  test_labels = np.fromfile(f, dtype='uint8', count=-1, sep='')

with open('test-images.txt', 'rb') as f:
  test_images = np.fromfile(f, dtype='bool', count=-1, sep='').reshape((num_examples,) + (28, 28, 1))

```

For visualizing the images (requires additional dependency `matplotlib`):

```python
import matplotlib.pyplot as plt

plt.imshow(train_images[0], cmap='binary')
plt.show()
```
