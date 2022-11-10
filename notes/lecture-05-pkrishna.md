# Lecture 05: Quantization (Part I)

## Note Information

| Title       | Quantization (Part I)                                                                                           |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 09/22/2022                                                                                                      |
| Note Author | Pranav Krishna (pkrishna)                                                                                       |
| Description | The first half of an introduction to Quantization, which is a common model compression technique                |


## What is Quantization?

Quantization is the process of turning a continuous signal discrete, in a broad sense; it is common in signal processing (where you sample at discrete intervals) and image compression (where you reduce the space of possible colors for each pixel). This technique is *orthogonal* to pruning (from the last two lectures).


## Numeric Data Types

In Machine Learning, Quantization involves changing the data type of each weight to a more restrictive data type (i.e. can be represented in less bits). This section briefly describes the common data types used in Machine Learning models/

### Integers

Integers can be either signed or unsigned. If they are unsigned, then they are just $n$-bit numbers in the range $[0, 2^n-1]$. If they are signed, there are two ways they can be represented.
* *Signed-Magnitude*: represent the numbers $[-2^{n-1}-1, 2^{n-1}-1]$, where the first bit is a 'sign' bit, where $0$ is positive and $1$ is negative; then the rest of the $n-1$ bits represent the number. Its main drawback is that $1000\dots$ and $0000\dots$ represent the same number ($0$), among other quirks
* *Two's Complement*: represent the numbers $[-2^n, 2^{n-1}]$. Here, the idea is that $-x-1 = \sim x$ (bitwise NOT). Note that the negative numbers 'go the other way' compared to the previous representation type. It removes the redundancy of the previous method.

### Fixed Point Numbers

These represent decimals, but behave very similar to integers, in the sense that they are just integers but shifted by a power of $2$. They have a fixed number of bits to represent numbers before and after the decimal.

### Floating Point Numbers

This is a much more common method for representing real numbers. The data is split into three parts - a sign bit (usually the first), the exponent bits, and the mantissa/significant bits/fraction.

A number is then $(-1)^{\text{sign}} \times (1 + \text{mantissa}) + 2^{\text{exponent} - \text{exponent bias}}$, where the mantissa is read as a binary decimanl, and the exponent is read as an integer. If we let $k$ be the number of bits used to represent the exponent, then the exponent bias is defined as $2^{k-1}-1$.

Here are a list of Floating-Point Conventions and the number of bits used for each part:

| Convention                       | Sign Bits | Exponent Bits | Mantissa |
|----------------------------------|-----------|---------------|----------|
| IEEE 754                         | 1         | 8             | 23       |
| IEEE Half-Precision 16-bit float | 1         | 5             | 10       |
| Brain Float (BF16)               | 1         | 8             | 7        |
| NVIDIA TensorFloat 32            | 1         | 8             | 10       |
| AMD 24-bit Float (AMD FP24)      | 1         | 7             | 16       |

*Comments on systems*: The Brain Float trades range with accuracy and was specifically designed for the neural network setting, because precision is less important than range. For the NVIDIA - the '32' does not refer to the total number of bits (which is 19), but instead this refers to the number of exponent bits used (8), which is the same as IEEE 754. 19 bits may seem like a weird number, but NVIDIA uses specialized hardware for this to be optimal.

## K-Means Based Weight Quantization

**Motivation:** As hinted at with Brain Float, Neural Network performance actually does not depend much on the precision of the weights. In general, values like 2.09, 2.12, 1.92, and 1.87 can all be approximated as 2 just fine. So - why don't we approximate the weights as such?

**Method:** Cluster the weights into $n$ buckets (using the K-Means Clustering Algorithm), where $n$ is generally chosen to be some power of $2$. Then, approximate each weight that appears in the network with the mean of all the weights that have been placed in the bucket. Store the mapping of bucket ids to values in a *codebook*, and for each weight store the bucket id.

### Use

During inference, you would then replace the bucket ids with the weights that they represent according to the codebook, and do normal floating-point operations. If you want to finetune the model, then you would just perform gradient descent with the parameters being the codebook entries. In general - add the gradients of each weight in the bucket to get the gradient for the entire bucket.

Another important thing to note is that if you want to both prune and quantize a model, the general practice is to *prune first, quantize second*.

### Analysis

In terms of the storage compression ratio, let $N$ be the original number of bits per weight, $n$ be the number of buckets used for clustering, and $W$ be the number of weights. Then, the compression ratio is approximately $\frac{W\log_2n + Nn}{WN}$; the second term is for the codebook storage. In general, as the model size gets bigger (i.e $W \rightarrow \infty$), the compression ratio approaches $\frac{\log_2n}{N}$.

For inference time, it might seem counterintuitive that it speeds up, as the codebook scheme adds another layer to the computation - fetching the values using the bucket ids. However, the speedup can be found when considering the actual architecture. In general, because of the size compression, there will be less costly DRAM accesses and data-transfers into cache. Note that this codebook does require a specialized architecture to achieve the full potential of the method - the values in the codebook are generally stored in on-chip SRAM, which takes one cycle to access.

### Pushing the Limits

**Hyperparameter Selection**: The only hyperparameter to select for this method is the number of codebook entries - in general, 16 is used as standard practice; experiments with AlexNet show that Convolutional Layers need 4 bits and Fully-Connected Layers need 2 bits before a significant drop in accuracy is hit. Curiously, these thresholds are independent of whether or not the model was pruned first.

**SqueezeNet**: In general, one may ask: why do we not train compressed models to begin with? The creators of SqueezeNet decided to give this a try. Coupled with Pruning and Quantization, the model achieved a ridiculous 510x compression rate while maintaining both the Top-1 and Top-5 accuracy values of AlexNet. A picture of the architecture is provided:

**Huffman Encoding**: We make an implicit assumption that all the buckets must be represented by the same number of bits. But, evidently, some buckets are going to appear much more frequently than others. This begs the question - why don't we use a lower amount of bits for more frequent weights? Using a Huffman Encoding for this does the trick, and can improve the compression ratio even further.

## Linear Quantization

*Content Warning from Song: There is math involved.*

The second main type of Quantization is Linear Quantization - where you want an affine map from integers to a real-numbered interval.
$$r = S(q - z)$$
In this equation, $S$ is the scale factor, and is generally a floating-point number; $q$ is the quantized version of $r$, and therefore an integer; and $Z$ is an integer of the same type that is chosen such that it maps to zero.

### Finding the Values (Asymmetric Quantization)

Let $r_\min, r_\max$ be the minimum and maximum of all the original weights; and let $q_\min, q_\max$ be the minimum and maximum of the quantized range (which is generally $-2^n, 2^n-1$ for some $n$). Then, we would approximately want $r_\min = S(q_\min - Z)$ and $r_\max = S(q_\max - Z)$. Subtracting the second from the first gives $r_\max - r_\min = S(q_\max - q_\min)$, which gives us:
$$S = \frac{r_\max - r_\min}{q_\max - q_\min}$$

For the zero offset, we would like our quantization scheme to be able to represent zero exactly. So, even though from the equation $r_\min = S(q_\min - Z)$, we could get $Z = q_\min - \frac{r_\min}{S}$, we change this to $Z = \text{round}\left(q_\min - \frac{r_\min}{S}\right)$.

### Symmetric Quantization

The first scheme covered was asymmetric quantization, named for the inherent asymmetry between the positive and negative values it can represent. Another scheme that can be used is symmetric quantization; in this case, the $Z$ value is fixed at $0$, while the new scaling factor becomes $S = \frac{\lvert r \rvert_\max}{-q_{\min}}$, using similar notation to the previous section.

While the implementation is easier, and the logic for handling zero is cleaner, this causes the quantized range to be wasted effectively (i.e. there are a range of values that can be represented by our scheme but do not need to be); this is especially true after any ReLU operation, in which we know the values are to be nonnegative, which essentially loses a whole bit of information! In general, this means we don't use this scheme to quantify activations, but we could for quantifying weights

### The Math

Now, suppose we have a linear layer (with no bias) in our model - $Y = WX$; with our quantization scheme, this becomes:
    $Y = WX$ \\
    $S_Y(q_Y - Z_Y) = S_W(q_W - z_W) \cdot S_X(q_X - Z_x)$ \\
    $q_Y = \frac{S_WS_X}{S_Y}(q_W - z_W)(q_X - Z_X) + Z_Y$ \\
    $q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X - z_Wq_X - Z_Xq_W + Z_WZ_X) + Z_Y$ \\

Note here that the term $Z_Xq_W + Z_WZ_X$ can be precomputed, as this is not dependent on the specific input ($Z_X$ is dependent on only our quantization scheme, determined through global statistics across the input dataset); and, we can set $Z_W = 0$ via choosing a symmetric quantization scheme.

#### Adding Bias

If we add bias, with $b = S_b(q_b - Z_b)$, we should also set $Z_b = 0$ to match $W$, and to make computations simpler, we can just make the scaling factor $S_WS_X$ (this works well in practice). Then, we would have:

  $Y = WX + b$ \\
  $S_Y(q_Y - Z_Y) = S_W(q_W - z_W) \cdot S_X(q_X - Z_x) + S_b(q_b - Z_b)$ \\
  $q_Y = \frac{S_WS_X}{S_Y}(q_W - z_W)(q_X - Z_X) + Z_Y + \frac{S_b}{S_Y}(q_b - Z_b)$ \\
  $q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X - z_Wq_X - Z_Xq_W + Z_WZ_X + q_b - Z_b) + Z_Y$ \\
  $q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X + q_b - Z_Xq_W) + Z_Y$ \\
  $q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X + q_{\text{bias}}) + Z_Y$ \\

where we define $q_{\text{bias}} = q_b - Z_Xq_W$, as this can be precomputed, through a similar argument as before.

#### Convolutions

It turns out that, because convolutions are also essentially a linear operator, the derivation for its quantization is extremely similar to that for a linear layer. Through similar definitions (i.e. $Z_W = Z_b = 0, S_b = S_WS_X$), we would get $q_Y = \frac{S_WS_X}{S_Y}\left(\text{Conv}(q_W, q_X) + q_{\text{bias}}\right) + Z_Y$, where $q_{\text{bias}} = q_b - \text{Conv}(q_W, Z_X)$.
