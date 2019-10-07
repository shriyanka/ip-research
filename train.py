from data import *
import model
import os

import tensorflow as tf

WEIGHT_DECAY = 0.0005
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
TRAIN_EPOCHS = 45
INPUT_SIZE = 100
LEARNING_RATE = 0.0001
LOG_DIR = "C:/Users/kajal/PycharmProjects/image-processing-research/person-segmentation/logs/"
DATA_ROOT = 'C:/Users/kajal/research-dataset/person_cropped/'

train_dataset, test_dataset = create_dataset(
      os.path.join(DATA_ROOT, 'train_i/*.jpg'),
      os.path.join(DATA_ROOT, 'val_i/*.jpg'),
      INPUT_SIZE, TRAIN_BATCH_SIZE)

net = model.Model(weight_decay=WEIGHT_DECAY)

def loss(logits, labels):
    x = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
    return x

def evaluate_model(net, val_ds):
    val_ds_iterator = val_ds
    mean_loss = 0.0
    counter = 0
    for (img, gt) in val_ds_iterator:
        logits = net(img, is_training=False)
        #print("val", gt.shape, logits.shape)
        try:
            mean_loss += loss(logits, gt)
        except Exception as e:
            pass
        counter += 1
    mean_loss /= counter
    return mean_loss

optimizer = tf.optimizers.Adam(LEARNING_RATE)
writer = tf.summary.create_file_writer(LOG_DIR)
global_step = tf.compat.v1.train.get_or_create_global_step()
ckpt_prefix = os.path.join(LOG_DIR, 'ckpt')
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=net, gs=global_step)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_prefix, max_to_keep=5)
if ckpt_manager.latest_checkpoint is not None:
    print('Restoring from checkpoint: {}'.format(ckpt_manager.latest_checkpoint))
    ckpt.restore(ckpt_manager.latest_checkpoint)

for (img, gt) in train_dataset:
    #print("train", img.shape, gt.shape)
    gs = global_step.numpy()

    # Forward
    with tf.GradientTape() as tape:
        logits = net(img, is_training=True)
        loss_value = loss(logits, gt) + sum(net.losses)

    # Backward
    grads = tape.gradient(loss_value, net.variables)
    optimizer.apply_gradients(zip(grads, net.variables))

    # Display loss and images
    if gs % 100 == 0:
        #with writer.as_default():
            #tf.summary.scalar('loss', loss_value)
            #tf.summary.image('seg', tf.concat([gt, tf.sigmoid(logits)], axis=2))
        print(gs, "Training Loss: %2.4f" % (tf.reduce_mean(loss_value)))

    # Calc validation loss
    if gs % 200 == 0:
        val_loss = evaluate_model(net, test_dataset)
        #with writer.as_default():
            #tf.summary.scalar('val_loss', tf.reduce_sum(val_loss))
        print("[%4d] Validation Loss: %2.4f" % (gs, tf.reduce_mean(val_loss)))

    # Save checkpoint
    if gs % 1000 == 0:
        ckpt_manager.save()

ckpt_manager.save()
