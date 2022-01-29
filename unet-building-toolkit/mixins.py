import tensorflow as tf
import abc


class ConvolutionMixin(abc.ABC):
    @abc.abstractmethod
    def apply_convolutions(self, input_data, convolutions, filters, batch_normalization, **kwargs):
        pass


class ConvMixin2D(ConvolutionMixin):
    def apply_convolutions(self, input_data, convolutions, filters, batch_normalization, **kwargs):
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution2D(filters, (3, 3), padding='same')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class ConvMixin3D(ConvolutionMixin):
    def apply_convolutions(self, input_data, convolutions, filters, batch_normalization, **kwargs):
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution3D(filters, (3, 3, 3), padding='same')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class AttentionMixin(abc.ABC):
    @abc.abstractmethod
    def add_attention_gate(self):
        pass


class AttentMixin2D(AttentionMixin):
    pass


class AttentMixin3D(AttentionMixin):
    pass