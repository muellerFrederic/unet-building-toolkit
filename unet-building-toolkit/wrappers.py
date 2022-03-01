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
