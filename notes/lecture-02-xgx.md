# Lecture 02: Basics of Neural Networks

## Note Information

| Title       | Basics of Neural Networks                                                                |
| ----------- | ---------------------------------------------------------------------------------------- |
| Lecturer    | Song Han                                                                                 |
| Date        | 09/13/2022                                                                               |
| Note Author | Guangxuan Xiao (xgx)                                                                     |
| Description | Review the basics of deep learning and introduce efficiency metrics for neural networks. |

## Terminology of neural networks

Synapses = Weights = Parameters: Connection intensities in neural nets.

Neurons  = Features = Activations: Input / intermediate / output values in neural nets.

## Popular neural network layers

### Fully-connected layer (linear layer)

The output neuron is connected to all input neurons.

$y_i = \sum_j w_{ij}x_j+b_i$, where $n$ is the batch size, $c_i$ is the number of input channels, and $c_o$ is the number of output channels.

| Tensors             | Shapes       |
| ------------------- | ------------ |
| Input Features $X$  | $(n, c_i)$   |
| Output Features $Y$ | $(n, c_o)$   |
| Weights $W$         | $(c_o, c_i)$ |
| Bias $b$            | $(c_o, )$    |

### Convolution layer

The output neuron is only connected to input neurons in the receptive field.

| Tensors             | Shapes of 1-D Conv | Shapes of 2-D Conv     |
| ------------------- | ------------------ | ---------------------- |
| Input Features $X$  | $(n, c_i, w_i)$    | $(n, c_i, h_i, w_i)$   |
| Output Features $Y$ | $(n, c_o, w_o)$    | $(n, c_o, h_o, w_o)$   |
| Weights $W$         | $(c_o, c_i,k_w)$   | $(c_o, c_i, k_h, k_w)$ |
| Bias $b$            | $(c_o, )$          | $(c_o, )$              |

- Multiplications needed to calculate a single output element equals the size of the convolution kernels.

- How to calculate output height and width?
  
  $h_o = h_i - k_h + 1$

#### Padding

Padding keeps the output feature map size the same as the input feature map size.

- Zero padding pads the input boundaries with zero.
  
  $h_o = h_i + 2p-k_h + 1$

- Other: reflection padding, replication padding, constant padding.

#### Receptive field

- In convolution, each output element depends on $k_h \times k_w$ in the input.

- Each successive convolution adds $k-1$ to the receptive field size.

- With L layers the receptive field size is $L (k-1) + 1$.

#### Strided convulution layer

Strided convulution layers increase the receptive field without increasing the layer.

$h_o = \frac{h_i + 2p-k_h}{s}+1$, where s is the stride and p is the padding.

#### Group convolution layer

A **Grouped Convolution** uses a group of convolutions - multiple kernels per layer - resulting in multiple channel outputs per layer. This leads to wider networks helping a network learn a varied set of low level and high level features.

Weights W: $(c_o, c_i, k_h, k_w) \rightarrow (g \times c_o/g, c_i/g, k_h, k_w)$  

#### Depthwise convolution layer

Independent filter for each channel: $g= c_i = c_o$ in grouped convolution.

Weights: $(c, k_h, k_w)$

#### Other convolution layers

- Transposed convolution can be used to upsample the feature map (e.g., in super-resolution and segmentation).

- Dilated convolution can be used to increase the receptive field with dilated convolution kernels.

### Pooling layer

Pooling layers are used to downsample the feature map and gradually reduce the size of feature maps.

Usually, the stride is the same as the kernel size.

### Normalization layer

Normalizing the features makes the model optimization faster

- The normalization layer normalized the features as follows
  
  $\hat{x}_i = \frac{x_i-\mu_i}{\sigma}$,
  
  $\mu_i = \frac{1}{m}\sum_{i\in\delta_i}x_k$,
  
  $\sigma_i=\sqrt{\frac{1}{m}\sum_{k\in\delta_i}(x_k-\mu_i)^2+\varepsilon}$

### Activation function

Activation functions are usually non-linear.

- ReLU is efficient, and quantization-friendly, but has zero gradients for negative inputs.

- GeLU, Swish, and Hardswish are more effective but hard to be quantized.

### Transformers

Transformer is composed of multiple self-attention layers, which find the relationship between tokens. The computational complexity of self-attention is $O(n^2)$, where $n$ is the token size. Therefore, it will be very slow with large token size.

## Popular CNN architectures

- AlexNet [[Krizhevsky *et al.*, 2012]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) is the first architecture to adopt an architecture with consecutive convolutional layers (conv layer 3, 4, and 5).

- ResNet [[He *et al.*, 2015]](https://arxiv.org/abs/1512.03385) introduce a residual connection to allow gradient flow.

- MobileNetV2 [[Sandler *et al.*, 2018]](https://arxiv.org/abs/1801.04381) introduces depthwise convolution to effectively reduce the computation cost.

## Efficiency metrics for deep learning

The two core metrics are **Computation and Memory**.

Three dimensions of consideration are **Storage, Latency, and Energy**.

### Latency

Latency = $max(T_{operation}, T_{memory})$

### Energy consumption

Memory accesses are more energy-consuming than computations. Below is a ranking of energy consumption:

DRAM Access > SRAM Access > FP Mult > INT Mult > Register File > FP Add > INT Add

Therefore, we should avoid data movement, since the more data movement, the more memory references will lead to more energy consumption.

### Memory-related metrics

#### Number of parameters

- Linear: $c_0 \times c_i$

- Convolution: $c_o c_i k_h k_w$

- Grouped convolution: $c_o c_i k_h k_w / g$

- Depthwise convolution: $c_o k_w k_h$

#### Model size

- $Model Size = NumOfParameters \times BitWidth$

For example, AlexNet has 61M parameters, so its model size would be 244MB (with FP32) and 61MB (with INT8).

#### Total and peak activation

- Imbalanced memory distribution of MobileNet: the bottleneck is the peak (inference) and sum (training)

- Training bottleneck: not weight but activation

- MCUNet: Activation getting smaller, weights getting larger because the channel is increasing and the resolution is decreasing.

### Computation-related metrics

#### MACs: multiply-accumulate operations

A multiply-accumulate (MAC) operation is $a = a+b\times c$. Here is the number of MACs for some common operators:

- Matrix-vector multiplication (MV): $m\times n$

- General matrix-matrix multiplication (GEMM): $m\times n\times k$

- Linear layer: $c_o\times c_i$

- Convolution: $c_i\times k_w \times k_h \times h_o \times w_o \times c_o$

- Grouped convolution: $c_i\times k_w \times k_h \times h_o \times w_o \times c_o / g$

- Depthwise convolution: $k_w \times k_h \times h_o \times w_o \times c_o$

- As an example, AlexNet has 724M MACs.

#### FLOP: floating point operation

- One floating-point multiply-accumulate (MAC) operation corresponds to two floating-point operations (FLOP): multiply and add.

For instance, AlexNet has 724M MACs, corresponding to 1.4G FLOP.

- Floating point operation per second (FLOPS)
  
  $FLOPS = \frac{FLOP}{second}$
