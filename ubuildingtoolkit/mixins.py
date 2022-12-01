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
This module encapsulates mixins for different functionalities like applying a certain amount of convolution operations
to a given input or creating attention gates based on given inputs.
"""

import abc
import tensorflow as tf


class ConvolutionMixin(abc.ABC):
    """Interface for convolution mixins"""
    @abc.abstractmethod
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        pass


class ConvMixin2D(ConvolutionMixin):
    """Encapsulates the apply_convolutions method implemented for usage with 2-dimensional input data"""
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        """
        Applies a given amount of 2D convolution operations to the input data under the given constraints.
        :param input_data: keras tensor that should be transformed through a given amount of convolutions
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param kernel_size: the size of the kernel that should be used in the convolution operation
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       activation='elu', kernel_initializer='he_normal')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class ConvMixin3D(ConvolutionMixin):
    """Encapsulates the apply_convolutions method implemented for usage with 3-dimensional input data"""
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        """
        Applies a given amount of 3D convolution operations to the input data under the given constraints.
        :param input_data: keras tensor that should be transformed through a given amount of convolutions
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param kernel_size: the size of the kernel that should be used in the convolution operation
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution3D(filters=filters, kernel_size=kernel_size, padding='same',
                                                       activation='elu', kernel_initializer='he_normal')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class AttentionMixin(abc.ABC):
    """Interface for mixins which apply attention gates on given inputs"""
    @abc.abstractmethod
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        pass


class AttentMixin2D(AttentionMixin):
    """Encapsulates the add_attention_gate method implemented for usage with 2-dimensional input data"""
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        """
        Creates an attention gate based on the given data
        :param skip_connection_signal: signal from the contracting path of the network
        :param gating_signal: signal from the lower level of the expanding path
        :param filters: the amount of feature maps that should be created by each convolution
        """
        x_in = tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1), strides=(2, 2),
                                             padding='same')(skip_connection_signal)
        gating_signal = tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1),
                                                      padding='same')(gating_signal)
        output = tf.keras.layers.Add()([x_in, gating_signal])
        output = tf.keras.layers.ReLU()(output)
        output = tf.keras.layers.Convolution2D(filters=1, kernel_size=(1, 1), padding='same',
                                               activation='sigmoid')(output)
        output = tf.keras.layers.UpSampling2D()(output)
        output = tf.keras.layers.Multiply()([skip_connection_signal, output])
        return output


class AttentMixin3D(AttentionMixin):
    """Encapsulates the add_attention_gate method implemented for usage with 3-dimensional input data"""
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        """
        Creates an attention gate based on the given data
        :param skip_connection_signal: signal from the contracting path of the network
        :param gating_signal: signal from the lower level of the expanding path
        :param filters: the amount of feature maps that should be created by each convolution
        """
        x_in = tf.keras.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), strides=(2, 2, 2),
                                             padding='same')(skip_connection_signal)
        gating_signal = tf.keras.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1),
                                                      padding='same')(gating_signal)
        output = tf.keras.layers.Add()([x_in, gating_signal])
        output = tf.keras.layers.ReLU()(output)
        output = tf.keras.layers.Convolution3D(filters=1, kernel_size=(1, 1, 1), padding='same',
                                               activation='sigmoid')(output)
        output = tf.keras.layers.UpSampling3D()(output)
        output = tf.keras.layers.Multiply()([skip_connection_signal, output])
        return output
