import tensorflow as tf
import abc
import mixins


class Block(abc.ABC):
    @abc.abstractmethod
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        pass


class Residual2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Residual3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Standard2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        return self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)


class Standard3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        return self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)


class Dense2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3), batch_normalization)
        return concat_arr[-1]


class Dense3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3, 3), batch_normalization)
        return concat_arr[-1]


class Inception2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling2D(strides=(1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1), batch_normalization)


class Inception3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling3D(strides=(1, 1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1, 1), batch_normalization)


class StandardSkip2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3), batch_normalization)
        return output


class StandardSkip3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3, 3), batch_normalization)
        return output


class ResidualSkip2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3), batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class ResidualSkip3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3, 3), batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class DenseSkip2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        input_data = tf.keras.layers.Concatenate()(input_data)
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3), batch_normalization)
        return concat_arr[-1]


class DenseSkip3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        input_data = tf.keras.layers.Concatenate()(input_data)
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3, 3), batch_normalization)
        return concat_arr[-1]


class InceptionSkip2D(Block, mixins.ConvMixin2D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        input_data = tf.keras.layers.Concatenate()(input_data)
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling2D(strides=(1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1), batch_normalization)


class InceptionSkip3D(Block, mixins.ConvMixin3D):
    def create(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        input_data = tf.keras.layers.Concatenate()(input_data)
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling3D(strides=(1, 1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1, 1), batch_normalization)
