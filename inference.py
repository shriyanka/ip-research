import sys

import cv2
import imageio
import numpy as np

import tensorflow as tf

import model

CKPT_DIR = 'logs/ckpt/'
INPUT_IMG_FPATH = 'img/me_512.png'
OUTPUT_IMG_FPATH = 'img/me_512_seg.png'

net = model.Model()

ckpt = tf.train.Checkpoint(model=net)
ckpt.restore(tf.train.latest_checkpoint(CKPT_DIR))

img = cv2.imread(INPUT_IMG_FPATH, 0)
img = cv2.resize(img, (256, 256))
img = img[None, ...].astype(np.float32) / np.float32(255.)
img = np.stack((img,)*1, axis=-1)
logits = net(img, is_training=False)
print(img.shape, logits.shape)

img_out = tf.sigmoid(logits).numpy()
img_out = np.round(img_out[0, ...] * 255.).astype(np.uint8)
cv2.imwrite(OUTPUT_IMG_FPATH, img_out)
sys.exit(1)