import tensorflow as tf
import abc
import convolution_mixins


class Block(abc.ABC):
    @abc.abstractmethod
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        pass


class Residual2D(Block, convolution_mixins.Mixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Residual3D(Block, convolution_mixins.Mixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Standard2D(Block, convolution_mixins.Mixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        return self.apply_convolutions(input_data, convolutions, filters, batch_normalization)


class Standard3D(Block, convolution_mixins.Mixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        return self.apply_convolutions(input_data, convolutions, filters, batch_normalization)


class StandardSkip2D(Block, convolution_mixins.Mixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, batch_normalization)
        return output


class StandardSkip3D(Block, convolution_mixins.Mixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, batch_normalization)
        return output


class ResidualSkip2D(Block, convolution_mixins.Mixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class ResidualSkip3D(Block, convolution_mixins.Mixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output
