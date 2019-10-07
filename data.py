import os
import shutil

import cv2
import tensorflow as tf
import glob

IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load(image_file, type):
  """Loads the image and generates input and target image.

  Args:
    image_file: .jpeg file

  Returns:
    Input image, target image
  """

  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image, channels=3)

  if type == "train":
    anot = tf.convert_to_tensor(tf.strings.regex_replace(image_file, "train_i", "train_a"))
  else:
    anot = tf.strings.regex_replace(image_file, "val_i", "val_a")

  anot = tf.strings.regex_replace(anot, ".jpg", ".png")

  anot_im = tf.io.read_file(anot)
  anot_im = tf.image.decode_png(anot_im, channels=3)

  input_image = tf.cast(image, tf.float32)
  real_image = tf.cast(anot_im, tf.float32)

  return input_image, real_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width])
  real_image = tf.image.resize(real_image, [height, width])

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, tf.shape(input_image)[2]]) #change to 3 here

  return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function
def random_jitter(input_image, real_image):
  """Random jittering.

  Resizes to 286 x 286 and then randomly crops to IMG_HEIGHT x IMG_WIDTH.

  Args:
    input_image: Input Image
    real_image: Real Image

  Returns:
    Input Image, real image
  """
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


def load_image_train(image_file):
  input_image, real_image = load(image_file, "train")
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file, "val")
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def create_dataset(path_to_train_images, path_to_test_images, buffer_size,
                   batch_size):
  """Creates a tf.data Dataset.

  Args:
    path_to_train_images: Path to train images folder.
    path_to_test_images: Path to test images folder.
    buffer_size: Shuffle buffer size.
    batch_size: Batch size

  Returns:
    train dataset, test dataset
  """
  # files = glob.glob(path_to_train_images)
  # files
  train_dataset = tf.data.Dataset.list_files(path_to_train_images)
  train_dataset = train_dataset.shuffle(buffer_size)
  train_dataset = train_dataset.map(
      load_image_train, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.list_files(path_to_test_images)
  test_dataset = test_dataset.map(
      load_image_test, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size)

  return train_dataset, test_dataset

def checkImages():
  DATA_ROOT = 'C:/Users/kajal/research-dataset/person_cropped/train_a/*.png'
  files = glob.glob(DATA_ROOT)
  for file in files:
    im = cv2.imread(file)
    if im is not None:
      print(im.shape)
    else:
      print(im, file)
      os.remove(file)
      os.remove(file.replace("train_a", "train_i").replace("png", "jpg"))

checkImages()