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
This module contains wrapper-classes for applying standard up- or downsampling operations to given input data.
This is needed to adjust the interfaces of the underlying classes anD make them interchangeable.
"""

import tensorflow as tf
import abc


class SubSamplingBlock2D(abc.ABC):
    @abc.abstractmethod
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        pass


class TransposedConvolutionWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=kernel_size,
                                                      strides=strides)(input_data)


class UpSamplingWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.UpSampling2D(size=kernel_size)(input_data)


class MaxPoolingWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)(input_data)


class AveragePoolingWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)(input_data)


class DownConvolutionWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_data)


class MixedPoolingWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        max = tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)(input_data)
        max = tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1), padding='same')(max)
        avg = tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)(input_data)
        avg = tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1), padding='same')(avg)
        add = tf.keras.layers.Add()([max, avg])
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1), padding='same',
                                             activation='sigmoid')(add)


class SubSamplingBlock3D(abc.ABC):
    @abc.abstractmethod
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        pass


class TransposedConvolutionWrapper3D(SubSamplingBlock3D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=kernel_size,
                                                      strides=strides)(input_data)


class UpSamplingWrapper3D(SubSamplingBlock3D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.UpSampling2D(size=kernel_size)(input_data)


class MaxPoolingWrapper3D(SubSamplingBlock3D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)(input_data)


class AveragePoolingWrapper3D(SubSamplingBlock3D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)(input_data)


class DownConvolutionWrapper3D(SubSamplingBlock3D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_data)


class MixedPoolingWrapper3D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        max = tf.keras.layers.MaxPooling3D(pool_size=kernel_size, strides=strides)(input_data)
        max = tf.keras.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), padding='same')(max)
        avg = tf.keras.layers.AveragePooling3D(pool_size=kernel_size, strides=strides)(input_data)
        avg = tf.keras.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), padding='same')(avg)
        add = tf.keras.layers.Add()([max, avg])
        return tf.keras.layers.Convolution3D(filters=filters, kernel_size=(1, 1, 1), padding='same',
                                             activation='sigmoid')(add)
