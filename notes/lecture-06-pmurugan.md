# Efficient ML Notes 9/27

# Quantization Part II

### Administrative Info

Lab 2 Out, due Oct 18

# Review of K-Means and Linear Quantization

### K-Means

Have quantized weights with floating-point operations with lookup table

### Linear

Have integer quantized weights with affine map from real to integer weights

Map q_min and q_max to r_min and r_max, with zero point Z and scale S

[linear quantization assumptions]

Do bulk of operations in N-bit integer with 32-bit conversion for overflow

[Linear quantization example, asymmetric vs symmetric]

# Post-Training Quantization (PTQ)

## 1. Weight Quantization

$|r_{max}| = |W_{max}|$

Can use single scale $S$ for whole tensor (Per-Tensor Quantization). Works for large models, but poor accuracy for small models

Why? Because large differences in weights between channels and outlier weights can make quantization mapping not effective.

Solution is to use per-channel quantization.

[Per channel vs Per tensor quantization]

There are differences in quantized values (e.g. (2,3)) because scale factors are different. Leads to a lower reconstruction error for the quantized matrix.

The drawback of per-channel quantization is that it requires more specialized hardware support, and requires more computation and memory overhead because more scaling factors must be stored. However, modern GPUs are starting to support per-channel quantization.

Can we make weight ranges similar to each other so that per-tensor weight quantization will work?

### Weight Equalization

The key is to scale the layer $W^{(i)}_{oc=j}$ by a scale factor $s$ and the corresponding $W^{(i+1)}_{ic=j}$ by $s^{-1}$. To make weight ranges as close as possible throughout the model, we set [formula for s]

This naive implementation assumes that the activation between the two layers is linear (thus works best for ReLU). This approach also requires multiple passes to propagate changes throughout the model, but does not require retraining.

### Adaptive Rounding

Why Adaptive Rounding? The philosophy is that rounding to the nearest integer is not necessarily optimal. The best rounding scheme is the one that reconstructs the original activation the best.

[math]

Learn a tensor $V$ that chooses between rounding up or down for each weight element. Add regularization so that $V$ is encouraged to be binary.

## 2. Activation quantization

Need to collect activation statistics to effectively quantize activations

1. [EMA]
2. [Calibration batch]
Use the mean of the min and max of each sample to minimize effect of outliers.
Usually pick representative sample at random (e.g. 100 images)
1. Minimize the MSE between inputs and reconstructed quantized inputs, assuming a Gaussian or Laplace distribution (with parameters estimated from the input distribution)
2. Minimize loss of information, measured by KL divergence. Still need to identify intuitively where to clip large activations to minimize KL divergence. Widely used in TensorRT.

[Weight distribution clipping figure]

## 3. Bias quantization

A common assumption is that quantization error is unbiased, but according to [x study] this is not necessarily true.

Quantization error can induce bias in subsequent layers, shifting the input distribution of later layers. [quantization error eqns] Can absorb quantization error into bias, can estimate without additional calibration data from batch normalization parameters.

Smaller models seem to not respond as well to post training quantization, presumably because of their smaller representational capacity. Bias correction can help for smaller models.

# Quantization-Aware Training (QAT)

We want to propagate gradients through model and through quantization operations. For K-means, pool gradients for each centroid and update centroid value. Learning the index has minimal benefit, so it stays fixed through finetuning.

### Linear quantization

Maintain full-precision copy of the weights so that small gradients are maintained without loss of precisions, and simulate quantization for feed-forward steps. How do we compute gradients for full precision and quantized weights? Because the quantization functions are discrete-valued, the gradient is zero almost everywhere so the weights and activations don’t get activated. The simplest solution is to use a Straight-Through Estimator, where we assume the gradient of the quantization function is the identity operator.

[accuracies]

Area moving very quickly, on specialized snapdragon hardware can reach >70% accuracy in less than 1 ms

For an N-bit quantization, the memory usage goes as O(N) and computation goes as O(N^2); however, memory is usually much more expensive, so a balance has to be reached per-application.

# Binary and Ternary Quantization

Instead of integer quantization, we can use binary weights or outputs to decrease memory and operator costs.

### Binarization of Weights

Deterministic binarization is simply the sign function.

Stochastic binarization quantizes probabilistically. For example, Binary Connect uses a hard sigmoid function to map inputs to +/- 1. Is more computationally expensive and has more hardware overhead.

Can also introduce floating point scaling factor of the binarized matrix, which improves accuracy significantly.

### Binary Activations and Weights

Operations are significantly simplified, so that matrix operations are replaced with bit operations. (e.g. multiplication ⇒ XNOR)

The full operation can be written as sum(XNOR)*2 - N, modify with bit operations

Have modest decrease in error for large decreases in memory and computation

### Ternary Weight Networks (TWN)

Improve accuracy by using a symmetric ternary quantization [math]

Quantization threshold is set to 0.7 empirically. Scaling factor is L1 norm of nonzero elements.

Although ternary quantization requires 2 bits, the symmetry and zero centroid enables implementation with less complex hardware multipliers.

### Trained Ternary Quantization (TTQ)

Can learn positive and negative scale factors of quantization. Further recovers accuracy.

# Mixed-Precision Quantization

Becoming more popular, now supported by some NVidia GPUs.

Can have different precisions for each layer, enables better accuracy and more quantized layers. However, has a large design space. Can use automated quantization methods with Actor-Critic models to optimize quantization factors.

Depthwise vs Pointwise bit allocations