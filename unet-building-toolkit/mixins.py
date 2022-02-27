import tensorflow as tf
import abc


class ConvolutionMixin(abc.ABC):
    @abc.abstractmethod
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        pass


class ConvMixin2D(ConvolutionMixin):
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution2D(filters, kernel_size, padding='same')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class ConvMixin3D(ConvolutionMixin):
    def apply_convolutions(self, input_data, convolutions, filters, kernel_size, batch_normalization, **kwargs):
        for x in range(convolutions):
            input_data = tf.keras.layers.Convolution3D(filters, kernel_size, padding='same')(input_data)
            if batch_normalization:
                input_data = tf.keras.layers.BatchNormalization()(input_data)
        return input_data


class AttentionMixin(abc.ABC):
    @abc.abstractmethod
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        pass


class AttentMixin2D(AttentionMixin):
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        x_in = tf.keras.layers.Convolution2D(filters, (1, 1), strides=(2, 2), padding='same')(skip_connection_signal)
        gating_signal = tf.keras.layers.Convolution2D(filters, (1, 1), padding='same')(gating_signal)
        output = tf.keras.layers.Add()([x_in, gating_signal])
        output = tf.keras.layers.ReLU()(output)
        output = tf.keras.layers.Convolution2D(1, (1, 1), padding='same', activation='sigmoid')(output)
        output = tf.keras.layers.UpSampling2D()(output)
        output = tf.keras.layers.Multiply()([skip_connection_signal, output])
        return output


class AttentMixin3D(AttentionMixin):
    def add_attention_gate(self, skip_connection_signal, gating_signal, filters):
        x_in = tf.keras.layers.Convolution3D(filters, (1, 1, 1), strides=(2, 2, 2), padding='same')(skip_connection_signal)
        gating_signal = tf.keras.layers.Convolution3D(filters, (1, 1, 1), padding='same')(gating_signal)
        output = tf.keras.layers.Add()([x_in, gating_signal])
        output = tf.keras.layers.ReLU()(output)
        output = tf.keras.layers.Convolution3D(1, (1, 1, 1), padding='same', activation='sigmoid')(output)
        output = tf.keras.layers.UpSampling3D()(output)
        output = tf.keras.layers.Multiply()([skip_connection_signal, output])
        return output
