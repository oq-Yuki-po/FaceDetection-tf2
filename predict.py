import tensorflow as tf
import typer
from tensorflow.keras.models import load_model

from modules.util import load_yaml


def predict():

    label = {0: "without_mask", 1: "with_mask"}

    config = load_yaml("config.yaml")
    image_path = config["image_path"]
    typer.echo("image_path: {}".format(image_path))

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, config["input_shape"][0:2])
    image = tf.expand_dims(image, axis=0)

    using_tflite = config["using_tflite"]

    if using_tflite:
        typer.echo("using tflite")
        tflite_path = config["tflite_path"]
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        typer.echo("output_data: {}".format(output_data))
        typer.echo("label: {}".format(label[output_data.argmax()]))
    else:
        typer.echo("using h5")
        model_path = config["model_path"]
        model = load_model(model_path)
        pred = model.predict(image)
        typer.echo(pred)
        typer.echo(label[pred.argmax()])


if __name__ == "__main__":
    typer.run(predict)
