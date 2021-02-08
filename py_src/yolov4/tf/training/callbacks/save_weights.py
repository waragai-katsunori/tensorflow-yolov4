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
from os import makedirs, path

from tensorflow.keras.callbacks import Callback


class SaveWeightsCallback(Callback):
    def __init__(
        self,
        yolo,
        dir_path: str = "trained-weights",
        step_per_save: int = 1000,
        weights_type: str = "tf",
    ):
        super().__init__()
        self._yolo = yolo
        self._weights_type = weights_type
        self._step_per_save = step_per_save

        makedirs(dir_path, exist_ok=True)

        self._path_prefix = path.join(dir_path, self._yolo.config.model_name)

        if weights_type == "tf":
            self.extension = "-checkpoint"
        else:
            self.extension = ".weights"

    def on_train_batch_end(self, batch, logs=None):
        step = self.model._train_counter

        if step % self._step_per_save == 0:
            self._yolo.save_weights(
                "{}-{}-step{}".format(
                    self._path_prefix, step.numpy(), self.extension
                ),
                weights_type=self._weights_type,
            )

    def on_train_end(self, logs=None):
        self._yolo.save_weights(
            "{}-final{}".format(self._path_prefix, self.extension),
            weights_type=self._weights_type,
        )
