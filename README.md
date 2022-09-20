# FaceMaskDetection

## project description

This project builds a deep learning model to determine if a face mask is worn.  
The model to be built is lightweight and converted to a tflite format for edge devices.

## dataset preparation

Prepare a dataset for mask detection from Kaggle or another appropriate source.
For example [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) from Kaggle.  
The dataset should be structured as follows:

```
dataset
├── train
│   ├── with_mask
│   │   ├── 0-with-mask.jpg
│   │   ├── 1-with-mask.jpg
│   │   ├── ...
│   ├── without_mask
│   │   ├── 0-without-mask.jpg
│   │   ├── 1-without-mask.jpg
│   │   ├── ...
├── test
│   ├── with_mask

```

convert the dataset to tfrecord format using the following command:

```
python modules/dataset_converter.py --dataset-path data --output-path data/tfrecord --division_count 10
```

## model

The model used for this project is the [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
and [MobileNetV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large) model from the [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) module.

## configuration

The configuration file is located in the `config` file.
The configuration file is a yaml file and contains the following parameters:

```
# train
data_path: "./data/tfrecord"
input_shape: [112, 112, 3]
backbone_type: "MobileNetV3Large"
batch_size: 32
epochs: 20

# predict
model_path: "./models/mobilenetv3large.h5"
image_path: "./test/image/without_mask.png"
```

## training

To train the model, run the following command:

```
python train.py
```

output folder structure:

```
outputs
├── 20210501_120000
│   ├── model
│   │   ├── model.h5
│   │   ├── model.tflite
│   ├── tensorboard
│   │   ├── train
│   │   │   ├── events.out.tfevents.1620000000.000000
│   │   ├── validation
│   │   │   ├── events.out.tfevents.1620000000.000000
│   ├── config.yaml
```

## prediction

To predict the model, run the following command:

```
python predict.py
```

output:

```
Model loaded
image_path: ./test/data/without_mask.png
[[1.000000e+00 7.350202e-23]]
without_mask
```
