from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split


@dataclass
class DataSet:

    path: List[str]
    image_size: Tuple[int, int] = (112, 112)
    batch_size: int = 32

    def __post_init__(self):
        self.train_path, self.validation_path = train_test_split(self.path, test_size=0.2)

    def _parse_tfrecord(self, is_training):
        def parse_tfrecord(tfrecord):
            features = {'with_mask': tf.io.FixedLenFeature([], tf.int64),
                        'encoded_image': tf.io.FixedLenFeature([], tf.string)}

            x = tf.io.parse_single_example(tfrecord, features)

            x_train = tf.image.decode_png(x['encoded_image'], channels=3)
            y_train = tf.cast(x['with_mask'], tf.int64)

            if is_training:
                x_train = self._transform_images()(x_train)

            y_train = self._transform_targets(y_train)
            return (x_train), y_train
        return parse_tfrecord

    def _transform_images(self):
        def transform_images(x_train):
            x_train = tf.image.resize(x_train, self.image_size)
            x_train = tf.image.random_flip_left_right(x_train)
            x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
            x_train = tf.image.random_brightness(x_train, 0.5)
            x_train = tf.cast(x_train, tf.float32) / 255.0

            return x_train
        return transform_images

    def _transform_targets(self, y_train):
        return y_train

    def load_tfrecord_dataset(self,
                              is_training: bool = True,
                              buffer_size=1024):
        """load dataset from tfrecord"""
        if is_training:
            raw_dataset = tf.data.TFRecordDataset(self.train_path)
            raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
        else:
            raw_dataset = tf.data.TFRecordDataset(self.validation_path)
        # raw_dataset = raw_dataset.repeat(-1)
        dataset = raw_dataset.map(self._parse_tfrecord(is_training),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
