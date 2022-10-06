# Lecture 05: Quantization (Part 1)

## Note Information

| Title       | Quantization (Part 1)                                                    |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 09/22/2022                                                                                                      |
| Note Author | Aaron Langham (alangham)        |                                                                                
| Description | Review of data types in computer systems, introduction to quantization in neural networks, and introduction to three common quantization methods. |

## Motivation
- Memory (and movement in memory) is expensive (in terms of energy consumption). [[Horowitz, M., IEEE ISSCC 2014]](https://ieeexplore.ieee.org/document/6757323)
- Lower bit-width operations are cheaper.
- We actually don't need all of the precision high bit-width offers anyway, so we can make 
deep learning more efficient by using lower bit-widths.

## Numeric Data Types

How do we represent numeric data in computer systems?

### Integer
- The most basic numeric data type, and can represent either unsigned or signed data.
- Unsigned integers with width $n$ can cover a range of $[0, 2^n - 1]$.
- Signed integers can convey their sign in two different ways:
    - Sign-Magnitude representation reserves the first bit for the sign, allowing a range of 
$[-2^{n-1} - 1, 2^{n-1} - 1]$. However, both all 0s and 1 followed by 0s represent the number 
0, which wastes one value.
    - Two's Complement representation covers a range of $[-2^{n-1}, 2^{n-1} - 1]$, with no 
wasted values. All 0s represents 0, and 1 followed by 0s represents $-2^{n-1}$.

### Fixed-Point Number
- Very similar to an integer, but a set of bits are reserved as the fraction.

### Floating-Point Number
- Unlike an integer, this data type does not have uniform spacing between values. However, it 
can cover a much larger range (important for NN training since gradients have large dynamic ranges).
- A 32-bit floating-point number specified by IEEE 754 consists of a sign bit, 8 exponent 
bits, and 23 fraction bits.
- The decimal value stored is the following:
$$(-1)^{\text{sign}} \times (1 + \text{fraction}) \times 2^{\text{exponent - 127}}$$
- Many floating-point specifications exist, such as:
  - IEEE FP32, IEEE FP16
  - Brain Float (BF16)
  - Nvidia TensorFloat (TF32)
  - AMD 24-bit Float (AMD FP24)
  
## Quantization
We define quantization as the process of constraining an input from a continuous or otherwise large set of values to a discrete or smaller set.

### K-Means-based Weight Quantization [[Han et al., ICLR 2016]](https://arxiv.org/pdf/1510.00149v5.pdf)
- Choose a cluster index bit-width, $B$.
- Run K-Means clustering on the weight matrix to find $2^B$ clusters.
- Store matrix of $B$-bit integer indexes (same size as weight matrix) and $B$-length "codebook" of centroids (in floating-point precision).
- To fine-tune trained model, group gradient elements into the stored centroids, sum each element for each centroid, multiply by the learning rate, and subtract from the stored centroids.
- Since weights occur with different frequencies, Huffman coding can be used to optimally encode the weights for fast access.
- However, all computation and memory access is still floating-point.

### Linear Quantization [[Jacob et al. CVPR 2018]](https://arxiv.org/pdf/1712.05877.pdf)
- Find the two parameters ($S$ and $Z$) that create an affine mapping of integers ($q$) to real numbers ($r$), such that:
$$r = S(q-Z)$$
- $Z$ sets the zero point, and is stored as an integer.
- $S$ sets the scale factor, and is stored as a floating-point number.
- For matrices, the floating-point computation $$\mathbf{Y} = \mathbf{W}\mathbf{X}$$ can be converted to the following quantized form:
$$\mathbf{q_Y} = \frac{S_W S_X}{S_Y} (\mathbf{q_W}\mathbf{q_X} - \mathbf{Z_W}\mathbf{q_X} - \mathbf{Z_X}\mathbf{q_W} + \mathbf{Z_W} \mathbf{Z_X}) + \mathbf{Z_Y}$$
- We can approximate $\mathbf{Z_W}$ as 0, since the trained weights tend to be zero-mean, and we can precompute $\mathbf{Z_X}\mathbf{q_W}$.
- Quantization can either be symmetric (at the cost of one quantization slot) or asymmetric (requires more complex implementation).
- The matrix multiplication given previously can be extended with a bias term ($\mathbf{Y} = \mathbf{W}\mathbf{X} + \mathbf{b}$) for the following:
$$S_Y (\mathbf{q_Y} - \mathbf{Z_Y}) = S_W S_X (\mathbf{q_W}\mathbf{q_X} - \mathbf{Z_X}\mathbf{q_W} + \mathbf{q_b} )$$
- This allows fully-connected layers and convolutional layers to compute with integers rather than floating-point precision.
