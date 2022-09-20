import tensorflow as tf
import typer
from tensorflow.keras.models import load_model

from modules.util import load_yaml


def predict():

    label = {0: "without_mask", 1: "with_mask"}

    config = load_yaml("config.yaml")
    model_path = config["model_path"]
    model = load_model(model_path)

    typer.echo("Model loaded")

    image_path = config["image_path"]
    typer.echo("image_path: {}".format(image_path))

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, config["input_shape"][0:2])
    image = tf.expand_dims(image, axis=0)

    pred = model.predict(image)

    typer.echo(pred)

    typer.echo(label[pred.argmax()])


if __name__ == "__main__":
    typer.run(predict)
