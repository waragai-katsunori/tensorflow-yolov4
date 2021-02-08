"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf


def save_as_tflite(
    model,
    tflite_path: str,
    quantization: str = "",
    dataset=None,
    num_calibration_steps: int = 100,
):
    """
    Save model and weights as tflite

    Usage:
        save_as_tflite(model=yolo.model, tflite_path="yolov4.tflite")
        save_as_tflite(
            model=yolo.model,
            tflite_path="yolov4-float16.tflite",
            quantization="float16"
        )
        save_as_tflite(
            model=yolo.model,
            tflite_path="yolov4-int.tflite",
            quantization="int",
            dataset=dataset,
            num_calibration_steps=200
        )
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    _supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    def representative_dataset_gen():
        count = 0
        while True:
            for images, _ in dataset:
                for i in range(len(images)):
                    yield [tf.cast(images[i : i + 1, ...], tf.float32)]
                    count += 1
                    if count >= num_calibration_steps:
                        return

    if quantization != "":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int":
        converter.representative_dataset = representative_dataset_gen
    elif quantization == "full_int8":
        converter.experimental_new_converter = False
        converter.representative_dataset = representative_dataset_gen
        _supported_ops += [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
    else:
        raise ValueError(f"'{quantization}' is not a valid quantization")

    converter.target_spec.supported_ops = _supported_ops

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(tflite_path, "wb") as fd:
        fd.write(tflite_model)
