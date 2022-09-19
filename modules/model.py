from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small


class FaceDetectionModel(Model):
    def __init__(self, backbone: str = "MobileNetV3Large"):
        super(FaceDetectionModel, self).__init__()
        if backbone == "MobileNetV3Large":
            self.backbone = MobileNetV3Large(input_shape=(112, 112, 3), include_top=False)
        elif backbone == "MobileNetV3Small":
            self.backbone = MobileNetV3Small(input_shape=(112, 112, 3), include_top=False)
        elif backbone == "MobileNetV2":
            self.backbone = MobileNetV2(input_shape=(112, 112, 3), include_top=False)
        self.pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(126, activation="relu")
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
