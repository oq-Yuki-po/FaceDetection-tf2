import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import typer
from tensorflow.keras import Model

from modules.dataset import DataSet
from modules.model import FaceMaskDetection
from modules.util import load_yaml


def train():

    # make output current datetime directory
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / now_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # load config
    config = load_yaml("config.yaml")
    backbone_type = config["backbone_type"]
    input_shape = tuple(config["input_shape"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    try:
        # copy config file to output directory
        shutil.copy("config.yaml", output_dir / "config.yaml")

        # load datasets
        p_tfrecords = list(Path(config["data_path"]).glob("*.tfrecord"))
        dataset = DataSet(path=[str(i) for i in p_tfrecords], image_size=input_shape[0:2], batch_size=batch_size)
        train_ds = dataset.load_tfrecord_dataset(is_training=True)
        val_ds = dataset.load_tfrecord_dataset(is_training=False)

        # build model
        face_detection_model = FaceMaskDetection(backbone_type=backbone_type, input_shape=input_shape)
        model = face_detection_model.build_model()
        model.compile(optimizer="adam",
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min'),
            tf.keras.callbacks.TensorBoard(log_dir=str(output_dir), histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(str(output_dir / "model.h5"),
                                               monitor='val_loss',
                                               save_best_only=True,
                                               mode='min')
        ]

        # train
        model.fit(train_ds,
                  epochs=epochs,
                  validation_data=val_ds,
                  callbacks=callbacks)

        model.fit(train_ds, epochs=10, validation_data=val_ds)

        # load best model
        best_model = tf.keras.models.load_model(str(output_dir / "model.h5"))

        # convert to tflite
        face_detection_model.convert_tflite(best_model, str(output_dir / "model.tflite"))

    except Exception as e:
        print(e)
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    typer.run(train)
