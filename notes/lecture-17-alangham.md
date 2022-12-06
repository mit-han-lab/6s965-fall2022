# Lecture 17: TinyEngine - Efficient Training and Inference on Microcontrollers

## Note Information

| Title       | TinyEngine - Efficient Training and Inference on Microcontrollers                                                    |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 11/08/2022                                                                                                      |
| Note Author | Aaron Langham (alangham)        |                                                                                
| Description | Introduction to microcontrollers, considerations in neural network deployment on microcontrollers, and the according optimization techniques. |

## Introduction to Microcontrollers
- Microcontrollers (MCUs) are low-cost, low-power miniature computers, typically without an operating system, with a huge variety of applications.
- MCUs typically have low computational power, limited memory, and a smaller instruction set.

## Neural Networks on Microcontrollers
- A key challenge is that MCU memory is often too small to hold deep neural networks (DNNs).
- Storing DNNs requires storing weights and activations.
  - Weights are static (after training), and thus can be stored in read-only flash memory.
  - Activations are dynamic (during inference), and thus should be stored in writeable SRAM (static random-access memory).

### Primary Data Layouts in Neural Networks
- Convolutional neural networks operate on four-dimensional tensors (feature maps N, channels C, kernel height H, and kernel width W).
- However, the order in storing the indices of this tensor is a design choice.
- From left to right, the order in which tensor elements are placed in contiguous memory:
  - NCHW (good for depthwise convolution)
  - NHWC (good for pointwise convolution)
  - CHWN (rarely used)

## Optimization Techniques in TinyEngine
- TinyEngine is designed to enhance computing speed and reduce memory usage for neural networks on MCUs [[Lin et al., NeurIPS 2020]](https://arxiv.org/abs/2007.10319), [[Lin et al., NeurIPS 2022]](https://arxiv.org/abs/2206.15472)

### Loop unrolling
- Convert loops to sequential statements
- Avoids the overhead of branching and loop control at the expense of a larger binary size
- Loops can be unrolled by a designable factor

### Loop reordering
- Improve data locality so that contiguous chunks of memory are accessed for each operation
- Avoid cache misses

### Loop tiling
- If data is much larger than cache size, it will be evicted and there will be a cache miss
- Convert loop into nested loop of a size determined by the cache size

### SIMD (Single Instruction Multiple Data) Programming
- Parallelize additions and multiplies at the data level by vectorizing the instruction

### Im2Col Convolution
- Convolution (which involves a 4D tensor) is reframed as a general matrix multiplication problem
- Can be faster, but requires additional memory space

### In-place Depth-wise Convolution
- Since channels are operated on independently with depth-wise convolution, perform activation operation in-place with a temporary buffer
- Almost 2x reduction in peak memory usage

### Appropriate Data Layout
- Use NHWC for point-wise convolution
- Use NCHW for depth-wise convolution

### Winograd Convolution (not implemented in TinyEngine)
- Instead of computing convolution, transform tensor, perform pointwise multiplication, then transform back
- Pointwise multiplication much faster than convolution, but this results in higher memory usage
