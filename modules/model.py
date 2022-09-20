from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small, mobilenet_v2, mobilenet_v3


class FaceMaskDetection():
    def __init__(self,
                 backbone_type: str = "MobileNetV3Large",
                 input_shape: Tuple[int, int, int] = (112, 112, 3)):
        self.input_shape = input_shape
        self.backbone_type = backbone_type
        if backbone_type == "MobileNetV3Large":
            self.backbone = MobileNetV3Large(input_shape=self.input_shape, include_top=False, weights="imagenet")
        elif backbone_type == "MobileNetV3Small":
            self.backbone = MobileNetV3Small(input_shape=self.input_shape, include_top=False, weights="imagenet")
        elif backbone_type == "MobileNetV2":
            self.backbone = MobileNetV2(input_shape=self.input_shape, include_top=False, weights="imagenet")
        else:
            raise ValueError(f"{self.backbone_type} is not supported.")
        self.inputs = layers.Input(shape=self.input_shape)
        self.pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(126, activation="relu")
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(2, activation='softmax')

    def build_model(self):
        inputs = self.inputs
        if self.backbone_type == "MobileNetV3Large":
            x = mobilenet_v3.preprocess_input(inputs)
        elif self.backbone_type == "MobileNetV3Small":
            x = mobilenet_v3.preprocess_input(inputs)
        elif self.backbone_type == "MobileNetV2":
            x = mobilenet_v2.preprocess_input(inputs)
        x = self.backbone(inputs)
        x = self.pool(x)
        x = self.dense(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return Model(inputs=inputs, outputs=outputs)

    def convert_tflite(self, model, ouput_path: str):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(ouput_path, "wb") as f:
            f.write(tflite_model)
