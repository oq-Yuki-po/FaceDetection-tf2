from pathlib import Path

from modules.dataset import DataSet


def test_dataset():
    p_tfrecords = list(Path("./data/tfrecord").glob("*.tfrecord"))
    train_dataset = DataSet(path=[str(i) for i in p_tfrecords], is_training=False)
    train_ds = train_dataset.load_tfrecord_dataset()
    for i in train_ds.take(1):
        print(i)
