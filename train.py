from pathlib import Path

from modules.dataset import DataSet
from modules.model import FaceDetectionModel


def train():

    p_tfrecords = list(Path("./data/tfrecord").glob("*.tfrecord"))

    dataset = DataSet(path=[str(i) for i in p_tfrecords])
    train_ds = dataset.load_tfrecord_dataset(is_training=True)
    val_ds = dataset.load_tfrecord_dataset(is_training=False)
    model = FaceDetectionModel()

    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, epochs=10, validation_data=val_ds)

if __name__ == "__main__":
    train()
