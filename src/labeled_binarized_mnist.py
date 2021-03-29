"""labeled_binarized_mnist dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
from tensorflow_datasets.image_classification import mnist
from six.moves import urllib
import numpy as np

# Markdown description that will appear on the catalog page.
_DESCRIPTION = """
"""

# BibTeX citation
_CITATION = """
"""

_URL = "https://github.com/JeroenHoekstra91/Binarized-MINST/raw/master/dataset/"

_TRAIN_IMAGES_FILENAME = "train-images.txt"
_TRAIN_LABELS_FILENAME = "train-labels.txt"
_TEST_IMAGES_FILENAME  = "test-images.txt"
_TEST_LABELS_FILENAME  = "test-labels.txt"

_TRAIN_EXAMPLES = 60000
_TEST_EXAMPLES  = 10000

class LabeledBinarizedMnist(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for labeled_binarized_mnist dataset."""

  VERSION       = tfds.core.Version('1.0.0')
  RELEASE_NOTES = { '1.0.0': 'Initial release.' }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder     = self,
        description = _DESCRIPTION,
        citation    = _CITATION,
        homepage    = "https://github.com/JeroenHoekstra91/Binarized-MINST",
        features    = tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=mnist.MNIST_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=mnist.MNIST_NUM_CLASSES)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys = ('image', 'label'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    
    filenames = {
        "train_data":   _TRAIN_IMAGES_FILENAME,
        "train_labels": _TRAIN_LABELS_FILENAME,
        "test_data":    _TEST_IMAGES_FILENAME,
        "test_labels":  _TEST_LABELS_FILENAME,
    }
    files = dl_manager.download(
        { data_type: urllib.parse.urljoin(_URL, filename) for data_type, filename in filenames.items() }
    )

    return [
        tfds.core.SplitGenerator(
            name       = tfds.Split.TRAIN,
            gen_kwargs = dict(
                num_examples = _TRAIN_EXAMPLES,
                images_path  = files["train_data"],
                label_path   = files["train_labels"]
            )
        ),
        tfds.core.SplitGenerator(
            name       = tfds.Split.TEST,
            gen_kwargs = dict(
                num_examples = _TEST_EXAMPLES,
                images_path  = files["test_data"],
                label_path   = files["test_labels"]
            )
        )
    ]

  def _generate_examples(self, num_examples, images_path, label_path):
    """Generate Labeled Binarized MNIST examples as dicts.
    
    Args:
        num_examples (int): The number of example.
        data_path (str): Path to the data files
        label_path (str): Path to the labels
    
    Yields:
      Generator yielding the next examples
    """
    # with tf.io.gfile.GFile(label_path, "rb") as f:
    with open(label_path, 'rb') as f:
        labels = np.fromfile(f, dtype='uint8', count=-1, sep='')

    # with tf.io.gfile.GFile(images_path, "rb") as f:
    with open(images_path, 'rb') as f:
        images = (np.fromfile(f, dtype='uint8', count=-1, sep='')
                .reshape((num_examples,) + mnist.MNIST_IMAGE_SHAPE))

    for index, image in enumerate(images):
        record = { "image": image, "label": labels[index] }
        
        yield index, record
