from pathlib import Path

import cv2
import tensorflow as tf
import typer
from sklearn.model_selection import StratifiedKFold


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str: bytes, with_mask: int) -> tf.train.Example:
    # Create a dictionary with features that may be relevant.
    feature = {'with_mask': _int64_feature(with_mask),
               'encoded_image': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(dataset_path: str = './data/train',
         division_count: int = 10,
         output_path: str = './data/tfrecord'):

    if not Path(dataset_path).exists():
        raise ValueError('Please define valid dataset path.')

    p_image_path = list(Path(dataset_path).glob('**/*.png'))

    p_readable_image_path = [i for i in p_image_path if cv2.imread(str(i)) is not None]

    labels = [1 if i.parent.name == "with_mask" else 0 for i in p_readable_image_path]
    image_paths = [str(i) for i in p_readable_image_path]

    assert len(labels) == len(image_paths)

    skf = StratifiedKFold(n_splits=division_count, shuffle=True, random_state=42)

    for index, (_, test_index) in enumerate(skf.split(image_paths, labels)):
        test_image_path = [image_paths[i] for i in test_index]
        test_image_label = [labels[i] for i in test_index]

        with tf.io.TFRecordWriter(f'{output_path}/{index}_{len(test_image_path)}_tfecord.tfrecord') as writer:
            print(f'Writing tfrecord file {index}_{len(test_image_path)}_tfecord.tfrecord')
            for sample_image_path, sample_label in zip(test_image_path, test_image_label):
                tf_example = make_example(img_str=open(sample_image_path, 'rb').read(),
                                          with_mask=sample_label)
                writer.write(tf_example.SerializeToString())  # type: ignore


if __name__ == '__main__':
    try:
        typer.run(main)
    except SystemExit:
        pass
