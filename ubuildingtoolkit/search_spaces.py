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
This module contains kerastuner search spaces for u-net-architectures suited for binary and multiclass segmentation on
2- and 3-dimensional input data.
"""

import math
import kerastuner as kt
import creators
import losses
import feature_recognition_blocks
import sub_sampling_blocks
import tensorflow as tf
import metrics


class SearchSpace2DBinary(kt.HyperModel):
    downsampling = {
        "max_pooling": sub_sampling_blocks.MaxPoolingWrapper2D(),
        "average_pooling": sub_sampling_blocks.AveragePoolingWrapper2D(),
        "down_convolution": sub_sampling_blocks.DownConvolutionWrapper2D(),
        "hybrid": sub_sampling_blocks.MixedPoolingWrapper2D()
    }

    upsampling = {
        "transposed_convolution": sub_sampling_blocks.TransposedConvolutionWrapper2D(),
        "up_sampling": sub_sampling_blocks.UpSamplingWrapper2D()
    }

    blocks_contracting_path = {
        "standard": feature_recognition_blocks.Standard2D(),
        "residual": feature_recognition_blocks.Residual2D(),
        "dense": feature_recognition_blocks.Dense2D(),
        "inception": feature_recognition_blocks.Inception2D()
    }

    blocks_expanding_path = {
        "standard": feature_recognition_blocks.StandardSkip2D(),
        "residual": feature_recognition_blocks.ResidualSkip2D(),
        "dense": feature_recognition_blocks.DenseSkip2D(),
        "inception": feature_recognition_blocks.InceptionSkip2D()
    }

    def __init__(self, input_shape, filter_base, pre_processing=None, post_processing=None):
        super().__init__()
        self.input_shape = input_shape
        self.filter_base = filter_base
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.max_depth = self._infere_max_depth(input_shape)
        self.output_feature_maps = 1


    @staticmethod
    def _infere_max_depth(input_shape):
        small_edge = min(input_shape[0], input_shape[1])
        return math.floor(math.log(small_edge/4, 2) + 1)

    def build(self, hp):
        depth = hp.Int(name='depth', min_value=2, max_value=self.max_depth)
        num_convolutions = hp.Int(name='num_convolutions', min_value=1, max_value=3)
        downsampling = self.downsampling[
            hp.Choice(name='downsampling', values=["max_pooling", "average_pooling", "down_convolution", "hybrid"])
        ]
        upsampling = self.upsampling[
            hp.Choice(name='upsampling', values=["transposed_convolution", "up_sampling"])
        ]
        block_contracting = self.blocks_contracting_path[
            hp.Choice(name='block_contracting', values=["standard", "residual", "dense", "inception"])
        ]
        block_expanding = self.blocks_expanding_path[
            hp.Choice(name='block_expanding', values=["standard", "residual", "dense", "inception"])
        ]
        batch_normalization = hp.Boolean(name='batch_normalization')
        attention_gate = hp.Boolean(name='attention_gate')

        model = creators.GeneralUnet2D().create(input_shape=self.input_shape,
                                                filter_base=self.filter_base,
                                                depth=depth,
                                                num_convolutions=num_convolutions,
                                                output_feature_maps=self.output_feature_maps,
                                                pre_processing=self.pre_processing,
                                                post_processing=self.post_processing,
                                                downsampling=downsampling,
                                                upsampling=upsampling,
                                                block_contracting=block_contracting,
                                                block_expanding=block_expanding,
                                                batch_normalization=batch_normalization,
                                                attention_gate=attention_gate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=losses.DiceLoss(),
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')])
        return model


class SearchSpace3DBinary(kt.HyperModel):
    downsampling = {
        "max_pooling": sub_sampling_blocks.MaxPoolingWrapper3D(),
        "average_pooling": sub_sampling_blocks.AveragePoolingWrapper3D(),
        "down_convolution": sub_sampling_blocks.DownConvolutionWrapper3D(),
        "hybrid": sub_sampling_blocks.MixedPoolingWrapper3D()
    }

    upsampling = {
        "transposed_convolution": sub_sampling_blocks.TransposedConvolutionWrapper3D(),
        "up_sampling": sub_sampling_blocks.UpSamplingWrapper3D()
    }

    blocks_contracting_path = {
        "standard": feature_recognition_blocks.Standard3D(),
        "residual": feature_recognition_blocks.Residual3D(),
        "dense": feature_recognition_blocks.Dense3D(),
        "inception": feature_recognition_blocks.Inception3D()
    }

    blocks_expanding_path = {
        "standard": feature_recognition_blocks.StandardSkip3D(),
        "residual": feature_recognition_blocks.ResidualSkip3D(),
        "dense": feature_recognition_blocks.DenseSkip3D(),
        "inception": feature_recognition_blocks.InceptionSkip3D()
    }

    def __init__(self, input_shape, filter_base, pre_processing=None, post_processing=None):
        super().__init__()
        self.input_shape = input_shape
        self.filter_base = filter_base
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.max_depth = self._infere_max_depth(input_shape)
        self.output_feature_maps = 1


    @staticmethod
    def _infere_max_depth(input_shape):
        small_edge = min(input_shape[0], input_shape[1])
        return math.floor(math.log(small_edge/4, 2) + 1)

    def build(self, hp):
        depth = hp.Int(name='depth', min_value=2, max_value=self.max_depth)
        num_convolutions = hp.Int(name='num_convolutions', min_value=1, max_value=3)
        downsampling = self.downsampling[
            hp.Choice(name='downsampling', values=["max_pooling", "average_pooling", "down_convolution", "hybrid"])
        ]
        upsampling = self.upsampling[
            hp.Choice(name='upsampling', values=["transposed_convolution", "up_sampling"])
        ]
        block_contracting = self.blocks_contracting_path[
            hp.Choice(name='block_contracting', values=["standard", "residual", "dense", "inception"])
        ]
        block_expanding = self.blocks_expanding_path[
            hp.Choice(name='block_expanding', values=["standard", "residual", "dense", "inception"])
        ]
        batch_normalization = hp.Boolean(name='batch_normalization')
        attention_gate = hp.Boolean(name='attention_gate')

        model = creators.GeneralUnet3D().create(input_shape=self.input_shape,
                                                filter_base=self.filter_base,
                                                depth=depth,
                                                num_convolutions=num_convolutions,
                                                output_feature_maps=self.output_feature_maps,
                                                pre_processing=self.pre_processing,
                                                post_processing=self.post_processing,
                                                downsampling=downsampling,
                                                upsampling=upsampling,
                                                block_contracting=block_contracting,
                                                block_expanding=block_expanding,
                                                batch_normalization=batch_normalization,
                                                attention_gate=attention_gate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=losses.DiceLoss(),
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=2, name='mean_iou')])
        return model


class SearchSpace2DMulticlass(kt.HyperModel):
    downsampling = {
        "max_pooling": sub_sampling_blocks.MaxPoolingWrapper2D(),
        "average_pooling": sub_sampling_blocks.AveragePoolingWrapper2D(),
        "down_convolution": sub_sampling_blocks.DownConvolutionWrapper2D(),
        "hybrid": sub_sampling_blocks.MixedPoolingWrapper2D()
    }

    upsampling = {
        "transposed_convolution": sub_sampling_blocks.TransposedConvolutionWrapper2D(),
        "up_sampling": sub_sampling_blocks.UpSamplingWrapper2D()
    }

    blocks_contracting_path = {
        "standard": feature_recognition_blocks.Standard2D(),
        "residual": feature_recognition_blocks.Residual2D(),
        "dense": feature_recognition_blocks.Dense2D(),
        "inception": feature_recognition_blocks.Inception2D()
    }

    blocks_expanding_path = {
        "standard": feature_recognition_blocks.StandardSkip2D(),
        "residual": feature_recognition_blocks.ResidualSkip2D(),
        "dense": feature_recognition_blocks.DenseSkip2D(),
        "inception": feature_recognition_blocks.InceptionSkip2D()
    }

    def __init__(self, input_shape, filter_base, num_classes, pre_processing=None, post_processing=None):
        super().__init__()
        self.input_shape = input_shape
        self.filter_base = filter_base
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.max_depth = self._infere_max_depth(input_shape)
        self.output_feature_maps = num_classes


    @staticmethod
    def _infere_max_depth(input_shape):
        small_edge = min(input_shape[0], input_shape[1])
        return math.floor(math.log(small_edge/4, 2) + 1)

    def build(self, hp):
        depth = hp.Int(name='depth', min_value=2, max_value=self.max_depth)
        num_convolutions = hp.Int(name='num_convolutions', min_value=1, max_value=3)
        downsampling = self.downsampling[
            hp.Choice(name='downsampling', values=["max_pooling", "average_pooling", "down_convolution", "hybrid"])
        ]
        upsampling = self.upsampling[
            hp.Choice(name='upsampling', values=["transposed_convolution", "up_sampling"])
        ]
        block_contracting = self.blocks_contracting_path[
            hp.Choice(name='block_contracting', values=["standard", "residual", "dense", "inception"])
        ]
        block_expanding = self.blocks_expanding_path[
            hp.Choice(name='block_expanding', values=["standard", "residual", "dense", "inception"])
        ]
        batch_normalization = hp.Boolean(name='batch_normalization')
        attention_gate = hp.Boolean(name='attention_gate')

        model = creators.GeneralUnet2D().create(input_shape=self.input_shape,
                                                filter_base=self.filter_base,
                                                depth=depth,
                                                num_convolutions=num_convolutions,
                                                output_feature_maps=self.output_feature_maps,
                                                pre_processing=self.pre_processing,
                                                post_processing=self.post_processing,
                                                downsampling=downsampling,
                                                upsampling=upsampling,
                                                block_contracting=block_contracting,
                                                block_expanding=block_expanding,
                                                batch_normalization=batch_normalization,
                                                attention_gate=attention_gate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[metrics.FixedMeanIoU(num_classes=self.output_feature_maps, name='mean_iou')])
        return model


class SearchSpace3DMulticlass(kt.HyperModel):
    downsampling = {
        "max_pooling": sub_sampling_blocks.MaxPoolingWrapper3D(),
        "average_pooling": sub_sampling_blocks.AveragePoolingWrapper3D(),
        "down_convolution": sub_sampling_blocks.DownConvolutionWrapper3D(),
        "hybrid": sub_sampling_blocks.MixedPoolingWrapper3D()
    }

    upsampling = {
        "transposed_convolution": sub_sampling_blocks.TransposedConvolutionWrapper3D(),
        "up_sampling": sub_sampling_blocks.UpSamplingWrapper3D()
    }

    blocks_contracting_path = {
        "standard": feature_recognition_blocks.Standard3D(),
        "residual": feature_recognition_blocks.Residual3D(),
        "dense": feature_recognition_blocks.Dense3D(),
        "inception": feature_recognition_blocks.Inception3D()
    }

    blocks_expanding_path = {
        "standard": feature_recognition_blocks.StandardSkip3D(),
        "residual": feature_recognition_blocks.ResidualSkip3D(),
        "dense": feature_recognition_blocks.DenseSkip3D(),
        "inception": feature_recognition_blocks.InceptionSkip3D()
    }

    def __init__(self, input_shape, filter_base, num_classes, pre_processing=None, post_processing=None):
        super().__init__()
        self.input_shape = input_shape
        self.filter_base = filter_base
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.max_depth = self._infere_max_depth(input_shape)
        self.output_feature_maps = num_classes


    @staticmethod
    def _infere_max_depth(input_shape):
        small_edge = min(input_shape[0], input_shape[1])
        return math.floor(math.log(small_edge/4, 2) + 1)

    def build(self, hp):
        depth = hp.Int(name='depth', min_value=2, max_value=self.max_depth)
        num_convolutions = hp.Int(name='num_convolutions', min_value=1, max_value=3)
        downsampling = self.downsampling[
            hp.Choice(name='downsampling', values=["max_pooling", "average_pooling", "down_convolution", "hybrid"])
        ]
        upsampling = self.upsampling[
            hp.Choice(name='upsampling', values=["transposed_convolution", "up_sampling"])
        ]
        block_contracting = self.blocks_contracting_path[
            hp.Choice(name='block_contracting', values=["standard", "residual", "dense", "inception"])
        ]
        block_expanding = self.blocks_expanding_path[
            hp.Choice(name='block_expanding', values=["standard", "residual", "dense", "inception"])
        ]
        batch_normalization = hp.Boolean(name='batch_normalization')
        attention_gate = hp.Boolean(name='attention_gate')

        model = creators.GeneralUnet3D().create(input_shape=self.input_shape,
                                                filter_base=self.filter_base,
                                                depth=depth,
                                                num_convolutions=num_convolutions,
                                                output_feature_maps=self.output_feature_maps,
                                                pre_processing=self.pre_processing,
                                                post_processing=self.post_processing,
                                                downsampling=downsampling,
                                                upsampling=upsampling,
                                                block_contracting=block_contracting,
                                                block_expanding=block_expanding,
                                                batch_normalization=batch_normalization,
                                                attention_gate=attention_gate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[metrics.FixedMeanIoU(num_classes=self.output_feature_maps, name='mean_iou')])
        return model
