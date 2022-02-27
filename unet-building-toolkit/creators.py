import abc
import blocks
import tensorflow as tf
import wrappers
import mixins


class ModelCreator(abc.ABC):
    @staticmethod
    def _create_filters(filter_base, depth):
        filters = [filter_base] * depth
        filters = [x * 2 ** count for count, x in enumerate(filters)]
        filters = filters + filters[:-1][::-1]
        return filters

    @staticmethod
    def _apply_processing(input_data, processing):
        if processing is None:
            return input_data
        for processing_step in processing:
            input_data = processing_step(input_data)
        return input_data

    @abc.abstractmethod
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        pass

    def create(self, input_shape, filter_base=64, depth=5, num_convolutions=2, output_feature_maps=2, pre_processing=None,
               post_processing=None, downsampling=None, upsampling=None, block_contracting=None, block_expanding=None,
               batch_normalization=False, **kwargs):
        network_input = tf.keras.layers.Input(shape=input_shape)
        filters = self._create_filters(filter_base=filter_base, depth=depth)
        network_structure = [None] * len(filters)
        network_structure[0] = self._apply_processing(network_input, pre_processing)
        network_structure = self._create_network_core(input_data=network_structure, filters=filters, depth=depth,
                                                      num_convolutions=num_convolutions, output_feature_maps=output_feature_maps,
                                                      downsampling=downsampling, upsampling=upsampling,
                                                      block_contracting=block_contracting,
                                                      block_expanding=block_expanding,
                                                      batch_normalization=batch_normalization, **kwargs)
        network_structure[-1] = self._apply_processing(network_structure[-1], post_processing)
        return tf.keras.Model(inputs=network_input, outputs=network_structure[-1])


class Unet(ModelCreator, mixins.AttentMixin2D):
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = wrappers.MaxPoolingWrapper2D()

        if upsampling is None:
            upsampling = wrappers.TransposedConvolutionWrapper2D()

        if block_contracting is None:
            block_contracting = blocks.Standard2D()

        if block_expanding is None:
            block_expanding = blocks.StandardSkip2D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                if 'attention_gate' in kwargs.keys() and kwargs['attention_gate']:
                    skip_connection = self.add_attention_gate(skip_connection, lower_stage, num_filters)
                upsampled = upsampling.apply_operation(filters=num_filters)(lower_stage)
                input_data[count] = block_expanding.create([skip_connection, upsampled], num_convolutions,
                                                           num_filters, **kwargs)

        input_data[-1] = tf.keras.layers.Convolution2D(output_feature_maps, (3, 3), padding='same')(input_data[-1])
        return input_data


class Unet3D(ModelCreator):
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = wrappers.MaxPoolingWrapper3D()

        if upsampling is None:
            upsampling = wrappers.TransposedConvolutionWrapper3D()

        if block_contracting is None:
            block_contracting = blocks.Standard3D()

        if block_expanding is None:
            block_expanding = blocks.StandardSkip3D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                upsampled = upsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_expanding.create([input_data[index_concatenate], upsampled], num_convolutions,
                                                           num_filters, **kwargs)

        input_data[-1] = tf.keras.layers.Convolution3D(output_feature_maps, (3, 3, 3), padding='same')(input_data[-1])
        return input_data


class Vnet(ModelCreator):
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        if downsampling is None:
            downsampling = wrappers.DownConvolutionWrapper2D()

        if upsampling is None:
            upsampling = wrappers.TransposedConvolutionWrapper2D()

        if block_contracting is None:
            block_contracting = blocks.Residual2D()

        if block_expanding is None:
            block_expanding = blocks.ResidualSkip2D()

        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                upsampled = upsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_expanding.create([input_data[index_concatenate], upsampled], num_convolutions,
                                                           num_filters, **kwargs)

        input_data[-1] = tf.keras.layers.Convolution2D(output_feature_maps, (3, 3), padding='same')(input_data[-1])
        return input_data


class GeneralUnet2D(ModelCreator, mixins.AttentMixin2D):
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                if 'attention_gate' in kwargs.keys() and kwargs['attention_gate']:
                    skip_connection = self.add_attention_gate(skip_connection, lower_stage, num_filters)
                upsampled = upsampling.apply_operation(filters=num_filters)(lower_stage)
                input_data[count] = block_expanding.create([skip_connection, upsampled], num_convolutions,
                                                           num_filters, **kwargs)

        input_data[-1] = tf.keras.layers.Convolution2D(output_feature_maps, (3, 3), padding='same')(input_data[-1])
        return input_data


class GeneralUnet3D(ModelCreator, mixins.AttentMixin3D):
    def _create_network_core(self, input_data, filters, depth, num_convolutions, output_feature_maps, downsampling, upsampling,
                             block_contracting, block_expanding, batch_normalization, **kwargs):
        for count, num_filters in enumerate(filters):
            if count == 0:
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            elif 0 < count < depth:
                input_data[count] = downsampling.apply_operation(filters=num_filters)(input_data[count - 1])
                input_data[count] = block_contracting.create(input_data[count], num_convolutions, num_filters, **kwargs)

            else:
                index_concatenate = 2 * depth - count - 2
                skip_connection = input_data[index_concatenate]
                lower_stage = input_data[count - 1]
                if 'attention_gate' in kwargs.keys() and kwargs['attention_gate']:
                    skip_connection = self.add_attention_gate(skip_connection, lower_stage, num_filters)
                upsampled = upsampling.apply_operation(filters=num_filters)(lower_stage)
                input_data[count] = block_expanding.create([skip_connection, upsampled], num_convolutions,
                                                           num_filters, **kwargs)

        input_data[-1] = tf.keras.layers.Convolution3D(output_feature_maps, (3, 3, 3), padding='same')(input_data[-1])
        return input_data
