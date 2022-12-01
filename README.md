# Unet Building Toolkit - ubuildingtoolkit
## About
The basic idea for this package originates in the field of hypertuning for neural network architectures which refers to tools and methods for finding the best possible combination of parameters which co-determine the structure of a neural network but cannot be learned.  
In such a context it is required to create and test various versions of the same architecture with just a few modifications so they can be compared and the best version can be chosen.  
Such a task can become really complex, costly and time-consuming if every version of the architecture needs to be implemented "by hand" using the tools and workflows provided with the standard frameworks for deep learning.  
And this is were this package comes in: Unet Building Toolkit allows you to create u-net-based architectures in tensorflow with a very simple api so it becomes way more practicable to create and compare diferent versions of the basic architecture so the best possible combination of hyperparameters can be found.  
U-Nets are pretty cool and useful networks used for semantic segmentaion. They are applied in medical tasks, material sciences and even in some industrial use cases just to name a few.

:warning:  
Note that this package has been tested manually to the best of my knowledge, but there are no automated tests.
Keep that in mind and use the package with care and think before you act.  
:warning:  
## Requirements & installation
Basically all you need to get started with ubuildingtoolkit is some version of Python above 3.8.  
You don't even need to install a specific version of tensorflow, since it is listed as dependency in the configuration files of this package.  
Ubuildingtoolkit has been (manually) tested with tensorflow 2.4.1.  
Install ubuildingtoolkit as follows:
```shell
pip install git+https://github.com/muellerFrederic/unet-building-toolkit.git
```
## Basics
The basic idea of ubuildingtoolkit is to split the original architecture into specific blocks which represent a specific task inside the architecture.  
For example there are some blocks for downsampling (MaxPooling in the original u-net).  
As long as the interface stays the same, these blocks can be exchanged and the architectures created can be compared and therefore optimized.  
A standard u-net suited for 2-dimensional inputs can be created as shown below:
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1))
```
### Blocks & Hyperparameters
To vary the architecure, pass the corresponding parameter to the .create method.  
#### Filterbase
Since the amount of filters is doubled up with every level in the u-net-architecture it is enough to pass the so-called filter-base to the method (which is exactly the amount of filters that will be used at the first level of the architecure).  
The code below creates a network with a filter-base of 20:
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1), filter_base=20)
```
#### Depth
Adjusting the depth of the network is pretty much the same as shown above for the filter-base:  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1), depth=3)
```
Please note that you can only go as deep as the dimensions of your input data allow MaxPooling without dropping below a 2x2x... tensor.  
#### Number of convolutions
This parameter refers to the amount of convolution operations used for feature processing at each level of the network (the original uses 2, but you can easilly go to 5 if you like to):  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1), num_convolutions=5)
```
#### Output feature maps
This parameter tells ubuildingtoolkit how many classes are annotated in the used dataset.  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1), output_feature_maps=5)
```
#### Downsampling method
This parameter controls which method is used for downsampling. Imagine you want to replace MaxPooling with AveragePooling:  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             downsampling=ubtk.sub_sampling_blocks.AveragePoolingWrapper2D())
```
Note that the dimensions of the operation need to fit the dimensions of the input data (Don't use 2d MaxPooling with 3d input data).  
The wrapper classes are needed to unify the interface of the operations and make them interchangeable.  
For example, the interface of the 2d MaxPooling operation looks like this: 
```python
tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)(input_data)
```
and the interface for convolutions looks like this (convolutions can be used for downsampling):
```python
tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_data)
```
A convolution requires the amount of kernels (filters) that should be used and MaxPooling does not.  
So we wrap both and unify the interface. The filters parameter is passed but dropped if the operation does not require it:  
```python
class MaxPoolingWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.MaxPooling2D(pool_size=kernel_size, strides=strides)(input_data)

class DownConvolutionWrapper2D(SubSamplingBlock2D):
    def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
        return tf.keras.layers.Convolution2D(filters=filters, kernel_size=kernel_size, strides=strides)(input_data)
```
#### Upsampling method
Everything mentioned above for the downsampling method also applies to the upsampling method in the u-net architecture.  
For example, you can replace the transposed convolution operation with an interpolation-based upsampling as follows:  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             upsampling=ubtk.sub_sampling_blocks.UpSamplingWrapper2D())
```
#### Feature recognition blocks
As said, the original u-net uses two convolutions per level to process the features contained in the input data.  
But there is a huge variety of methods to do so and some are included in this package.  
Ubuildingtoolkit includes the standard block with convolutions, residual blocks (basically a block which ads the original input to the processed input at the and of the processing), dense blocks (blocks that use multiple convolutions an concatenate the output of the previous blocks to the input of the current block) and inception blocks (which use parallel convolution and pooling operations).  
Every block is contained as a 2d- and 3d-version and suited for the contracting (downsampling) path and the expanding (upsampling) path of the network.  
Lets use residual blocks on the contracting path and inception blocks on the expanding path:  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             block_contracting=ubtk.feature_recognition_blocks.Residual2D(),
                             block_expanding=ubtk.feature_recognition_blocks.InceptionSkip3D())
```
Note that the block on the expanding path is called ...Skip2D/3D. This refers to the fact that the blocks on the expanding path of the u-net include the output of the corresponding level of the contracting path with a concatenation.  
#### Batch normalization
If you want to use batch normalization after each convolution operation, you can do this like so:  
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             batch_normalization=True)
```
#### Keyword arguments - **kwargs
There are basically two keyword arguments that you can pass to the .create method. One for the usage of attention gates and one for controlling the so-called growth-rate of the dense blocks.   
##### Attention gates
Attention gates add an additional processing step to the concatenation operation on the expanding path allowing the network to focus on "areas of interest" in the processed data.  
This means essentially that high activations on the contracting path are used to scale and combine the input to the concatenation operation on the contracting path.  
Use attention gates like this: 
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             attention_gates=True)
```
Note that attention gates are only available for the GeneralUnet classes. 
##### Growth rate for dense blocks
The growth rate of dense block determines how many convolution operations are put one after another and therefore how many outputs are concatenated before the data is put through the final operations.  
Modify the growth rate as follows:
```python
import ubuildingtoolkit as ubtk

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1),
                             growth_rate=6)
```
Note that the growth rate is only available for dense blocks. Passing the parameter will not lead to an error but will instead have no effect.  
#### Hypertuning
So finally, here is an example on hypertuning with ubuildingtoolkit.  
Assume that the best possible combination of the amount of convolutions and the depth of the network should be determined. Further we assume that load_dataset() returns the annotated data which the model should learn.    
You could loop over the parameters and compare the architectures (This hypertuning algorithm is also called grind search):
```python
import ubuildingtoolkit as ubtk

dataset_training = load_dataset(training)
dataset_testing = load_dataset(testing)
model_creator = ubtk.creators.GeneralUnet2D()
models_tested = []

for i in range(6):
    for j in range(2, 5):
        model = model_creator.create(input_shape=(128, 128, 1),
                                     depth=j,
                                     num_convolutions=i)
        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        model.fit(dataset_training, batch_size=16, epochs=20)
        evaluation_results = model.evaluate(dataset_testing, batch_size=16)
        models_tested.append({
            "depth": j,
            "num_convolutions": i,
            "metric_final": evaluation_results[1]
        })

models_tested.sort(key=lambda x: x["metric_final"])
best_model = models_tested[-1]
```
Note that this is more like pseudo-code and not a standalone example.
## Extending
If you would like to use a block that is not (yet) contained in the ubuildingtoolkit you can extend the package via inheritance.  
For example, if you have a super-secret downsampling method that you would like to include you can do this like so:
```python
import ubuildingtoolkit as ubtk

class MySuperSecretDownsamplingWrapper2D(ubtk.sub_sampling_blocks.SubSamplingBlock2D()):
        def apply(self, input_data, filters=None, kernel_size=(2, 2), strides=(2, 2)):
            return SuperSecretDownsampling2D(pool_size=kernel_size, strides=strides)(input_data)

model_creator = ubtk.creators.GeneralUnet2D()
model = model_creator.create(input_shape=(128, 128, 1), downsampling=MySuperSecretDownsamplingWrapper2D())
```
Note that SuperSecretDownsampling2D should be derived from keras Layer and Module like the already included methods.  
## To-dos
:round_pushpin: Figure out how to write tests for this package  
:round_pushpin: Add working examples  
:round_pushpin: Put the whole thing on PyPI  
## Contributing
If you would like to add something to the package, just send me a PR.
Explain what you did and why and why it adds value to the package.  
Be nice, thank you :relaxed:
