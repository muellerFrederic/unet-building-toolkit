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
This module contains classes which encapsulate methods for creating different u-net-like architectures.
"""

import abc
import tensorflow as tf
from . import feature_recognition_blocks
from . import sub_sampling_blocks
from . import mixins


class ModelCreator(abc.ABC):
    """
    The algorithm for creating a u-net-like structure is encapsulated in the abstract class.
    Parts of the algorithm need to be implemented in the derived classes -> template algorithm
    """
    @staticmethod
    def _create_filters(filter_base, depth):
        """
        The filters base is doubled up with each level of the network until the deepest level is reached.
        Then the amount of filters is divided by 2 with each level.
        :param filter_base: the filter base to use for the network structure
        :param depth: the amount of levels in the u-net-structure
        """
        filters = [filter_base] * depth
        filters = [x * 2 ** count for count, x in enumerate(filters)]
        filters = filters + filters[:-1][::-1]
        return filters

    @staticmethod
    def _apply_processing(input_data, processing):
        """
        This method takes a keras tensor an applies a given list of operation on it
        :param input_data: keras tensor that should be transformed
        :param processing: list of operations
        """
        if processing is None:
            return input_data
        for processing_step in processing:
            input_data = processing_step(input_data)
        return input_data

    @abc.abstractmethod
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        """
        This method needs to be implemented by the derived classes. The created network structure depends on the
        implementation of this method.
        :param input_data: keras tensor that should be processed through the network
        :param filters: a list containing the amount of filters that should be used at each level of the network
        :param depth: the amount of levels in the u-net-structure
        :param num_convolutions: the amount of convolutions that should be applied at each level of the u-net-structure
        :param output_feature_maps: the amount of classes on which the input should be mapped to
        :param downsampling: the downsampling method that should be used in the u-net-structure
        :param upsampling: the upsampling method that should be used in the u-net-structure
        :param block_contracting: the type of blocks from which the contracting path of the u-net-structure should be
        built.
        :param block_expanding: the type of blocks from which the expanding path of the u-net-structure should be
        built.
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        pass

    def create(self, input_shape, filter_base=64, depth=5, num_convolutions=2, output_feature_maps=2,
               pre_processing=None, post_processing=None, downsampling=None, upsampling=None, block_contracting=None,
               block_expanding=None, batch_normalization=False, **kwargs):
        """
        This method contains the actual algorithm for creating u-net-structures. Since the important parts are
        interchangeable, a broad variety of structures can be created.
        :param input_shape: the shape of the data that should be processed through the network
        :param filter_base: the amount of filters in the first level of the network
        :param depth: the amount of levels in the u-net-structure
        :param num_convolutions: the amount of convolutions that should be applied at each level of the u-net-structure
        :param output_feature_maps: the amount of classes on which the input should be mapped to
        :param pre_processing: list of transformations that should be applied before the actual network
        :param post_processing: list of transformations that should be applied after the actual network
        :param downsampling: the downsampling method that should be used in the u-net-structure
        :param upsampling: the upsampling method that should be used in the u-net-structure
        :param block_contracting: the type of blocks from which the contracting path of the u-net-structure should be
        built.
        :param block_expanding: the type of blocks from which the expanding path of the u-net-structure should be
        built.
        :param batch_normalization: wether to use batch_normalization after each convolution operation or not
        """
        network_input = tf.keras.layers.Input(shape=input_shape)
        filters = self._create_filters(filter_base=filter_base, depth=depth)
        network_structure = [None] * len(filters)
        network_structure[0] = self._apply_processing(network_input, pre_processing)
        network_structure = self._create_network_core(input_data=network_structure, filters=filters, depth=depth,
                                                      num_convolutions=num_convolutions,
                                                      output_feature_maps=output_feature_maps,
                                                      downsampling=downsampling, upsampling=upsampling,
                                                      block_contracting=block_contracting,
                                                      block_expanding=block_expanding,
                                                      batch_normalization=batch_normalization, **kwargs)
        network_structure[-1] = self._apply_processing(network_structure[-1], post_processing)
        return tf.keras.Model(inputs=network_input, outputs=network_structure[-1])


class Unet2D(ModelCreator, mixins.AttentMixin2D):
    """
    This class implements the _create_network_core method in a way, that it returns a standard u-net suited for
    2-dimensional input data.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = sub_sampling_blocks.MaxPoolingWrapper2D()

        if upsampling is None:
            upsampling = sub_sampling_blocks.TransposedConvolutionWrapper2D()

        if block_contracting is None:
            block_contracting = feature_recognition_blocks.Standard2D()

        if block_expanding is None:
            block_expanding = feature_recognition_blocks.StandardSkip2D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                upsampled = upsampling.apply(lower_stage, filters=num_filters)
                input_data[count] = block_expanding.apply([skip_connection, upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data


class Unet3D(ModelCreator):
    """
    This class implements the _create_network_core method in a way, that it returns a standard u-net suited for
    3-dimensional input data.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = sub_sampling_blocks.MaxPoolingWrapper3D()

        if upsampling is None:
            upsampling = sub_sampling_blocks.TransposedConvolutionWrapper3D()

        if block_contracting is None:
            block_contracting = feature_recognition_blocks.Standard3D()

        if block_expanding is None:
            block_expanding = feature_recognition_blocks.StandardSkip3D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                upsampled = upsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_expanding.apply([input_data[index_concatenate], upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data


class Vnet2D(ModelCreator):
    """
    This class implements the _create_network_core method in a way, that it returns a v-net suited for
    2-dimensional input data.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = sub_sampling_blocks.DownConvolutionWrapper2D()

        if upsampling is None:
            upsampling = sub_sampling_blocks.TransposedConvolutionWrapper2D()

        if block_contracting is None:
            block_contracting = feature_recognition_blocks.Residual2D()

        if block_expanding is None:
            block_expanding = feature_recognition_blocks.ResidualSkip2D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                upsampled = upsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_expanding.apply([input_data[index_concatenate], upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data


class Vnet3D(ModelCreator):
    """
    This class implements the _create_network_core method in a way, that it returns a v-net suited for
    3-dimensional input data.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = sub_sampling_blocks.DownConvolutionWrapper3D()

        if upsampling is None:
            upsampling = sub_sampling_blocks.TransposedConvolutionWrapper3D()

        if block_contracting is None:
            block_contracting = feature_recognition_blocks.Residual3D()

        if block_expanding is None:
            block_expanding = feature_recognition_blocks.ResidualSkip3D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                upsampled = upsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_expanding.apply([input_data[index_concatenate], upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data


class GeneralUnet2D(ModelCreator, mixins.AttentMixin2D):
    """
    This class implements the _create_network_core method in a way, that it needs every argument to be passed, since
    no standards are initialized when none. This allows to experiment with all the blocks and options that are
    implemented in this package.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                if 'attention_gates' in kwargs.keys() and kwargs['attention_gates']:
                    skip_connection = self.add_attention_gate(skip_connection, lower_stage, num_filters)
                upsampled = upsampling.apply(lower_stage, filters=num_filters)
                input_data[count] = block_expanding.apply([skip_connection, upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution2D(filters=output_feature_maps, kernel_size=(1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data


class GeneralUnet3D(ModelCreator, mixins.AttentMixin3D):
    """
    This class implements the _create_network_core method in a way, that it needs every argument to be passed, since
    no standards are initialized when none. This allows to experiment with all the blocks and options that are
    implemented in this package.
    """
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling,
                             upsampling, block_contracting, block_expanding, batch_normalization, **kwargs):
        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply(input_data[count - 1], filters=num_filters)
                input_data[count] = block_contracting.apply(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                if 'attention_gates' in kwargs.keys() and kwargs['attention_gates']:
                    skip_connection = self.add_attention_gate(skip_connection, lower_stage, num_filters)
                upsampled = upsampling.apply(lower_stage, filters=num_filters)
                input_data[count] = block_expanding.apply([skip_connection, upsampled], num_convolutions,
                                                          num_filters, **kwargs)

        if output_feature_maps == 1:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='sigmoid')(input_data[-1])
        else:
            input_data[-1] = tf.keras.layers.Convolution3D(filters=output_feature_maps, kernel_size=(1, 1, 1),
                                                           padding='same', activation='softmax')(input_data[-1])
        return input_data
