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

Neurons  = Features = Activations: Input / intermediate / Output values in neural nets.

## Popular Neural Network Layers

### Fully-connected layer (Linear layer)

The output neuron is connected to all input neurons.

$y_i = \sum_j w_{ij}x_j+b_i$

$n$: Batch size,

$c_i$: Input Channels

$c_o$: Output Channels

| Tensors             | Shapes       |
| ------------------- | ------------ |
| Input Features $X$  | $(n, c_i)$  |
| Output Features $Y$ | $(n, c_o)$   |
| Weights $W$         | $(c_o, c_i)$ |
| bias $b$            | $(c_o, )$    |

### Convolution Layer

The output neuron is connected to input neurons in the receptive field.

| Tensors             | Shapes of 1-D Conv | Shapes of 2-D Conv     |
| ------------------- | ------------------ | ---------------------- |
| Input Features $X$  | $(n, c_i, w_i)$    | $(n, c_i, h_i, w_i)$   |
| Output Features $Y$ | $(n, c_o, w_o)$    | $(n, c_o, h_o, w_o)$   |
| Weights $W$         | $(c_o, c_i,k_w)$   | $(c_o, c_i, k_h, k_w)$ |
| bias $b$            | $(c_o, )$          | $(c_o, )$              |

- Multiplications needed to calculate a single output element equals the size of the convolution kernels.

- How to calculate output height and width?

$h_o = h_i - k_h + 1$

#### Padding

Padding can keep the output feature map size the same as the  input feature map size.

- Zero padding pads the input boundaries with zero.
  
  $h_o + h_i + 2p-k_h + 1$

- Other: reflection padding, replication padding, constant padding.

#### Receptive Field

- In convolution , each output elelment depends on $k_h \times k_w$  in the input.

- Each successive convulution adds $k-1$ to the receptive field size.

- With L layers the receptive field size is $L (k-1) + 1$.

#### Strided Convulution Layer

Strided Convulution Layers increase the receptive field without increasing the layer.

$h_o = \frac{h_i + 2p-k_h}{s}+1$, where s is the stride and p is the padding.

#### Group Convolution Layer

A group of narrower convolutions.

Weights W: $(c_o, c_i, k_h, k_w) \rightarrow (g \times c_o/g, c_i/g, k_h, k_w)$  

#### Depthwise convolution layer

Independent filter for each channel: $g= c_i = c_o$ in grouped convolution.

Weights: $(c, k_h, k_w)$

#### Other convolution layers

- Transposed Convolution: super-resolution, segmentation, backprop.

- Dilated Convolution

### Pooling layer

Downsample the feature map to a smaller size, and graduallyreduceg the size of the feature maps.

Usually, the stride is the same as the kernel size.

### Normalization Layer

Normalizing the features makes training faster

- The normalization layer normalized the features as follows
  
  $\hat{x}_i = \frac{x_i-\mu_i}{\sigma}$,
  
  $\mu_i = \frac{1}{m}\sum_{i\in\delta_i}x_k$,
  
  $\sigma_i=\sqrt{\frac{1}{m}\sum_{k\in\delta_i}(x_k-\mu_i)^2+\varepsilon}$

### Activation function

Non-linear functions

- most efficient: ReLU

- ReLU problem: input lower than zero will have 0 gradient

- Quantization friendly: ReLU

- GeLU, Swish, hard swish: hard to quantize

### Transformers

Attention: finding the relationship between each token, problem $O(n^2)$ complexity and when $n$ is very large it will be very slow.

## Popular CNN Architectures

- AlexNet: the first architecture to adopt an architecture with consecutive convolutional layers (conv layer 3, 4, and 5).

- ResNet: introduce a residual connection to allow gradient flow.

- MobileNetV2: depth-wise convolution

## Efficiency metrics for Deep learning

Core: Computation and memory

Three Dimensions: Storage, Latency, and Energy

### Latency

Latency = $max(T_{operation}, T_{memory})$

### Energy consumption

Data movement -> more memory reference -> more energys

DRAM Access > SRAM Access > FP Mult > INT Mult > Register File > FP Add > INT Add

### Memory related

#### Number of Parameters

- Linear: $c_0 \times c_i$

- Convolution: $c_o c_i k_h k_w$

- Grouped Convolution: $c_o c_i k_h k_w / g$

- Depth-wise convolution: $c_o k_w k_h$

#### Model size

- $Model Size = NumOfParameters \times BitWidth$

- e.g. AlexNet 61M parameters, using fp32 = 244MB, using int8 = 61MB

#### Total / Peak activation

- Imbalanced memory distribution of MobileNet: the bottleneck is the peak (inference) and sum (training)

- Training bottleneck: not weight but activation

- MCUNet: Activation getting smaller, weights getting larger because the channel is increasing and the resolution is decreasing.

### Computation related

#### MACs: Multiply-Accumulate Operations

MAC: $a = a+b\times c$

- Matrix-vector multiplication (MV): $m\times n$

- General matrix-matrix multiplication (GEMM): $m\times n\times k$

- Linear layer: $c_o\times c_i$

- Convolution: $c_i\times k_w \times k_h \times h_o \times w_o \times c_o$

- Grouped Convolution: $c_i\times k_w \times k_h \times h_o \times w_o \times c_o / g$

- Depthwise Convolution: $k_w \times k_h \times h_o \times w_o \times c_o$

- AlexNet have 724MACs

#### FLOP: Floating Point Operation

- One floating-point MAC = Two FLOPs

e.g. AlexNet has 724 MACs, so the total number of floating point operations will be 1.4G FLOPs.

- Floating Point Operation Per Second (FLOPS)
  
  $FLOPS = \frac{FLOP}{second}$
