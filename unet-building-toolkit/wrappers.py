import tensorflow as tf
import abc


class AbstractUpSamplingWrapper(abc.ABC):
    @abc.abstractmethod
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        pass


class TransposedConvolutionWrapper(AbstractUpSamplingWrapper):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2))


class UpSamplingWrapper(AbstractUpSamplingWrapper):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.UpSampling2D(size=kernel_size)


class AbstractDownSamplingWrapper(abc.ABC):
    @abc.abstractmethod
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        pass


class MaxPoolingWrapper(AbstractDownSamplingWrapper):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)


class AveragePoolingWrapper(AbstractDownSamplingWrapper):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.AveragePooling2D(pool_size=kernel_size, strides=strides)


class DownConvolutionWrapper(AbstractDownSamplingWrapper):
    def apply_operation(self, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)


