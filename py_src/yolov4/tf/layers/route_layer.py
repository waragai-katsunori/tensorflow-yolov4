"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class RouteLayer(Layer):
    def __init__(self, metalayer, metanet):
        super().__init__(name=metalayer.name)
        self.metalayer = metalayer
        self.metanet = metanet

        groups = self.metalayer.groups
        group_id = self.metalayer.group_id

        def _split_and_select(x):
            return tf.split(x, groups, axis=-1)[group_id]

        def _route(x):
            return x

        def _concat(x):
            return K.concatenate(x, axis=-1)

        if self.metalayer.groups != 1:
            self._route_fucntion = _split_and_select
        else:
            if len(self.metalayer.layers) == 1:
                self._route_fucntion = _route
            else:
                self._route_fucntion = _concat

    def call(self, x):
        return self._route_fucntion(x)
