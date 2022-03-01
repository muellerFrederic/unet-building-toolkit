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
This is needed to adjust the interfaces of the underlying classes an make them interchangeable.
"""

import tensorflow as tf
import abc


class AbstractWrapper2D(abc.ABC):
    @abc.abstractmethod
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        pass


class TransposedConvolutionWrapper2D(AbstractWrapper2D):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides)


class UpSamplingWrapper2D(AbstractWrapper2D):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.UpSampling2D(size=kernel_size)


class MaxPoolingWrapper2D(AbstractWrapper2D):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)


class AveragePoolingWrapper2D(AbstractWrapper2D):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)


class DownConvolutionWrapper2D(AbstractWrapper2D):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)


class AbstractWrapper3D(abc.ABC):
    @abc.abstractmethod
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        pass


class TransposedConvolutionWrapper3D(AbstractWrapper3D):
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides)


class UpSamplingWrapper3D(AbstractWrapper3D):
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.UpSampling2D(size=kernel_size)


class MaxPoolingWrapper3D(AbstractWrapper3D):
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)


class AveragePoolingWrapper3D(AbstractWrapper3D):
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)


class DownConvolutionWrapper3D(AbstractWrapper3D):
    def apply_operation(self, filters=None, kernel_size=(2, 2, 2), strides=(2, 2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)
