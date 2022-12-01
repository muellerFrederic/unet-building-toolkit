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
This module contains building blocks of u-net-like architectures representing different methods of feature detection.
Since the interface is given as abstract class, all the blocks are interchangeable.
"""

import abc
import tensorflow as tf
from . import mixins


class FeatureRecognitionBlock(abc.ABC):
    """Interface for blocks representing different methods of feature detection"""
    @abc.abstractmethod
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        pass


class Standard2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Encapsulates a method for applying a standard block of convolution and batch normalization operations to
    2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a standard block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        return self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)


class Standard3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Encapsulates a method for applying a standard block of convolution and batch normalization operations to
    3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a standard block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        return self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)


class Residual2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Encapsulates a method for applying a residual block of convolution, batch normalization and add operations to
    2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a residual block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters. At the end the input data is added to the output of the block.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Residual3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Encapsulates a method for applying a residual block of convolution, batch normalization and add operations to
    3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a residual block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters. At the end the input data is added to the output of the block.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        repeats = filters // input_data.shape[-1]
        output = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        input_data = tf.tile(input_data, tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class Dense2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Encapsulates a method for applying a dense block of convolution, batch normalization and concatenation operations to
    2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a dense block. A dense block combines a given amount of standard blocks with concatenation operations.
        Each standard block gets the concatenated outputs of all standard blocks before as input.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        :param kwargs: pass the keyword argument growth_rate to determine the amount of standard blocks contained in the
        dense block. growth_rate=4 will lead to 4 standard blocks inside the dense block.
        """
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = input_data
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3), batch_normalization)
        return concat_arr[-1]


class Dense3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Encapsulates a method for applying a dense block of convolution, batch normalization and concatenation operations to
    3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a dense block. A dense block combines a given amount of standard blocks with concatenation operations.
        Each standard block gets the concatenated outputs of all standard blocks before as input.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        :param kwargs: pass the keyword argument growth_rate to determine the amount of standard blocks contained in the
        dense block. growth_rate=4 will lead to 4 standard blocks inside the dense block.
        """
        if 'growth_rate' in kwargs.keys():
            concat_arr = [0] * (kwargs['growth_rate'] + 1)
        else:
            concat_arr = [0] * 5
        concat_arr[0] = input_data
        concat_arr[1] = self.apply_convolutions(concat_arr[0], convolutions, filters, (3, 3, 3), batch_normalization)
        for x in range(2, len(concat_arr)):
            concat_arr[x] = tf.keras.layers.Concatenate()(concat_arr[:x])
            concat_arr[x] = self.apply_convolutions(concat_arr[x], convolutions, filters, (3, 3, 3), batch_normalization)
        return concat_arr[-1]


class Inception2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Encapsulates a method for applying an inception block to 2-dimensional input data.
    an inception block applies a 1x1, a 3x3, a 5x5 and a pooling/scaling operation in parallel to the given input data.
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates an inception block. Inside the inception block,  1x1,  3x3,  5x5 and a pooling/scaling operations are
        applied to the input data in parallel. Features of different sizes can therefore be detected at the same level
        of the network.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1), batch_normalization)


class Inception3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Encapsulates a method for applying an inception block to 3-dimensional input data.
    an inception block applies a 1x1, a 3x3, a 5x5 and a pooling/scaling operation in parallel to the given input data.
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates an inception block. Inside the inception block,  1x1x1,  3x3x3,  5x5x5 and a pooling/scaling operations
        are applied to the input data in parallel. Features of different sizes can therefore be detected at the same
        level of the network.
        :param input_data: keras tensor that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1, 1), batch_normalization)


class StandardSkip2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Skip-connection version of the Standard2D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a standard block of convolution and batch normalization operations to 2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a standard block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3), batch_normalization)
        return output


class StandardSkip3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Skip-connection version of the Standard3D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a standard block of convolution and batch normalization operations to 3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a standard block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3, 3), batch_normalization)
        return output


class ResidualSkip2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Skip-connection version of the Residual2D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a residual block of convolution, batch normalization and add operations to
    2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a residual block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters. At the end the input data is added to the output of the block.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3), batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class ResidualSkip3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Skip-connection version of the Residual3D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a residual block of convolution, batch normalization and add operations to
    3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a residual block containing a certain amount of convolution operations and optionally
        batch normalization by calling the apply_convolutions method of the convolution mixin class with the
        given parameters. At the end the input data is added to the output of the block.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        repeats = filters // input_data[1].shape[-1]
        output = tf.keras.layers.Concatenate()(input_data)
        output = self.apply_convolutions(output, convolutions, filters, (3, 3, 3), batch_normalization)
        input_data = tf.tile(input_data[1], tf.constant([1, 1, 1, 1, repeats]))
        output = tf.keras.layers.Add()([input_data, output])
        return output


class DenseSkip2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Skip-connection version of the Dense2D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a dense block of convolution, batch normalization and concatenation operations to 2-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a dense block. A dense block combines a given amount of standard blocks with concatenation operations.
        Each standard block gets the concatenated outputs of all standard blocks before as input.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        :param kwargs: pass the keyword argument growth_rate to determine the amount of standard blocks contained in the
        dense block. growth_rate=4 will lead to 5 standard blocks inside the dense block.
        """
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


class DenseSkip3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Skip-connection version of the Dense3D-block. Multiple inputs can be passed as list and will be concatenated.
    Applies a dense block of convolution, batch normalization and concatenation operations to 3-dimensional input data
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates a dense block. A dense block combines a given amount of standard blocks with concatenation operations.
        Each standard block gets the concatenated outputs of all standard blocks before as input.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        :param kwargs: pass the keyword argument growth_rate to determine the amount of standard blocks contained in the
        dense block. growth_rate=4 will lead to 5 standard blocks inside the dense block.
        """
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


class InceptionSkip2D(FeatureRecognitionBlock, mixins.ConvMixin2D):
    """
    Skip-connection version of the Inception2D-block. Multiple inputs can be passed as list and will be concatenated.
    Applying an inception block to 2-dimensional input data.An inception block applies a 1x1, a 3x3, a 5x5 and
    a pooling/scaling operation in parallel to the given input data.
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates an inception block. Inside the inception block,  1x1x1,  3x3x3,  5x5x5 and a pooling/scaling operations
        are applied to the input data in parallel. Features of different sizes can therefore be detected at the same
        level of the network.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        input_data = tf.keras.layers.Concatenate()(input_data)
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling2D(strides=(1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1), batch_normalization)


class InceptionSkip3D(FeatureRecognitionBlock, mixins.ConvMixin3D):
    """
    Skip-connection version of the Inception3D-block. Multiple inputs can be passed as list and will be concatenated.
    Applying an inception block to 3-dimensional input data.An inception block applies a 1x1, a 3x3, a 5x5 and
    a pooling/scaling operation in parallel to the given input data.
    """
    def apply(self, input_data, convolutions, filters=None, batch_normalization=True, **kwargs):
        """
        Creates an inception block. Inside the inception block,  1x1x1,  3x3x3,  5x5x5 and a pooling/scaling operations
        are applied to the input data in parallel. Features of different sizes can therefore be detected at the same
        level of the network.
        :param input_data: list of keras tensors that should be transformed
        :param convolutions: the amount of convolution operations that should be applied to the input data in each
        contained standard block
        :param filters: the amount of feature maps that should be created by each convolution
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        input_data = tf.keras.layers.Concatenate()(input_data)
        one_by_one = self.apply_convolutions(input_data, convolutions, filters, (1, 1, 1), batch_normalization)
        three_by_three = self.apply_convolutions(input_data, convolutions, filters, (3, 3, 3), batch_normalization)
        five_by_five = self.apply_convolutions(input_data, convolutions, filters, (5, 5, 5), batch_normalization)
        pooled = tf.keras.layers.MaxPooling3D(strides=(1, 1, 1), padding="same")(input_data)
        pooled = self.apply_convolutions(pooled, 1, filters, (1, 1, 1), batch_normalization)
        concat = tf.keras.layers.Concatenate()([one_by_one, three_by_three, five_by_five, pooled])
        return self.apply_convolutions(concat, 1, filters, (1, 1, 1), batch_normalization)
