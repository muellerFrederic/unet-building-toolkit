# Copyright 2022 Frédéric Müller
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This module contains loss functions that can be used to train a given model
"""

import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    """
    This class implements the dice- or f1-loss for tensorflow.
    Only suited for binary cases!
    """
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        num = 2 * tf.math.reduce_sum(y_true * y_pred)
        den = tf.math.reduce_sum(tf.math.square(y_true)) + tf.math.reduce_sum(tf.math.square(y_pred))
        return 1 - tf.reduce_mean(num/den)
