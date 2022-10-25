# Lecture 8: Neural Architecture Search (Part II)

## Note Information

| Title       | Introduction to TinyML and Efficient Deep Learning Computing                                                    |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 10/04/2022                                                                                                      |
| Note Author | Daniel Liu (dansl)                                                                                              |
| Description | Further deep dive into Neural Architecture Search, focusing on the evaluation aspect of architectures.          |

Recall that the general structure of Neural Architecture search consists of three parts: choosing an architecture search space, choosing a model from the architecture space, and evaluating the performance of that model for feedback and further iteration. 

In this lecture, we will be going over:
- The performance estimation strategies of NAS
- Include hardware awareness in the search space
- NAS as an application to other tasks.

## Performance Estimation

The simplest way to estimate the model quality is simply to train the model. However, this involves a lot of GPU hours to get the quality of a single model, which makes iteration infeasible. Thus, we need to find faster estimations for evaluating model quality.

## Weight Inheritance

One alternate way is to try **weight inheritance**. We inherit weights from pre-trained models to reduce the amount of time to fully train the model. Essentially, instead of training from scratch every time, we are borrowing weights from trained models and fine tuning for evaluation. For example, we can create two operations to modify a previous model: `Net2Wider` and `Net2Deeper`. 

`Net2Wider` increases the size of a particular layer by one, adding in an extra node. Then, we need to just fine-tune that specific node with the rest of the model.

`Net2Deeper` adds an extra layer. This is specifically done by adding a layer that is initialized as the identity linear transformation. Then, we need to just fine-tune that layer with the rest of the model. 

## HyperNetwork: Weight Generator

Instead, we can first sample the architecture randomly from the search space. HyperNetwork is a model that generates the weights based on what the architecture looks like. The error function is simply the amount of error from evaluating the model with HyperNetwork's generated weights, and a gradient descent method is used to update the HyperNetwork's weights to generate better weights in the future. 

## Performance Estimation Heuristics: ZenNAS

The idea of ZenNAS is that if we perturb a random input to the model, we expect the output to be sensitive to that perturbation. In practice, we calculate two perturbation metrics. 

The first is $z_1 = \log(f(x')-f(x))$, which is the log difference between the original output and the perturbed output.

The second is $z_2$, which is the log sum of the standard deviation of the inputs of each batch normalization layer.

## Performance Estimation Heuristics: GradSign

The idea of GradSign is that a good model should have the gradient sign agree for as many input activations as possible. Equivalently, the local minima of different input activations should be as close to each other as possible. The reason why we would like the gradients to agree is so when we train the model, the signs will agree and the model is efficient in training. 

## Hardware-Aware NAS

Previous NAS strategies we discussed like NASNet or DARTS are prohibitively expensive to run on standard GPUs. To avoid this problem, we have several **proxy tasks** we can consider optimizing instead to try to achieve a good model on the intended task:
- We can train the model on a simpler dataset that we hope is representative of progress on the intended task.
- We can try to minimize the architecture space as much as possible.
- We can use fewer epochs to train the model each iteration.
- We can try to minimize the number of FLOPs/parameters that the model has.
However, these methods all inherently have flaws.

## ProxylessNAS

Recall that for standard NASNet, when we are constructing a model, we try to add a particular layer design, train it, and if it's bad, then we throw it away and try gain. Instead, we can try to build a NAS network that trains the model with all possible layer designs attached to it, and choose the best one out of the choices. Essentially, we are testing out all the possible models with one pass of training instead of many iterations. 

This way of architecture search is much faster and negates the necessity of considering proxy tasks altogether. 

## Latency Increases

Just because we have fewer MACs, doesn't mean we have less latency. This depends on how well the model can do inference on the GPU. For example, if we add more layers to a model, the latency will increase, while if we add more hidden channels, then the latency won't actually increase as much. Furthermore, the specific hardware we're working with will affect what sort of operations will increase latency. 

How do we train a model to not increase too much latency on a piece of hardware? We could try to use it on the hardware itself, but this is slow. Instead, we can build a model to predict the latency that a model would have on a piece of hardware. A simple way of building such a latency predictor is to create a simple latency lookup table. Different types of layers will have different amounts of latency we can experimentally determine for a piece of hardware, so we can simply precompute the latencies in a table, then look up the latency when we need to evaluate a model later. Although primitive, this method is actually very accurate. 

However, sometimes, we can't experimentally get the latency of a particular layer, we can only experimentally get the latency of the entire model. In this case, we can train a neural network to take in the model specs as inputs, and try to predict the latency of the model as output. This method is also quite accurate for predicting actual hardware latency. 

As a result, with latency prediction models, we can train not only the weights of a model, but also change the architecture of the model itself every epoch, to optimize for the target device. 

But this means that for each different platform, we'll need to train a completely new model. Is it possible to train once to get an ideal model architecture for every single target platform, all at once? Turns out, the answer is yes.

## Improving Platform-Targeted Model Design Productivity

As previously mentioned, instead of training a network, getting the latency and accuracy, and iterating, we can try to train a single network, then select a certain sub-network with the best latency and accuracy. We can also repeat and choose the overall best sub-network. This idea is called **Once for All Network**.

Moreover, we can choose a different optimal subnetwork based on the target hardware, based on the computational capabilites of that hardware. This way, we don't need to train a new model every time we want to create a model for a specific piece of hardware. 

One concern is that the subnetworks may not perform very well. Fine tuning on the chosen subnetwork can be done to alleviate this. 

Surprisingly, we can get many sub-networks from a single Once for All network, over $10^{19}$.

## Progressive Pruning

If we want to create a sub-network with a smaller kernel size, depth, or channel width as we are training the model, we can use progressive pruning. We can reduce either the kernel size, depth, or channel width one step at a time. For example, we can reduce the depth by taking the output of the model before going through the last layer and using that as the final output. Or, we can reduce the channel width by analysing which channel contributes the least to the activations so far, and remove that channel.

## Roofline Analysis

The Once for All network paper uses **roofline analysis**. The basic idea is that since memory is expensive and computation is cheap, we want to maximize the number of operations per byte of memory loaded ($\text{OPS}/\text{Byte}$). 

## Neural Hardware Architecture Search

Instead of designing the model and trying to make it fit an existing hardware, can we design both the model architecture and the hardware design in parallel?

In model and hardware design, we in total have the following design choices:
- Accelerator (Architecture Sizing): Local Buffer Size, Global Buffer Size, \#PEs
- Accelerator (Connectivity Parameters): Compute Array Size, PE Connectivity
- Compiler: Loop Orders, Loop Tiling Size, Dataflow 
- Neural Network: \#Layers, \#Channels, Kernel Size, Bypass, (Input / Weight) Quantization Precision

## NAAS

NAAS does exactly this; it optimizes the design of the model in parallel with the hardware. This adds a respectable speedup to models without hardware architecture searching. Furthermore, it does better than humans in designing proper hardware fit for the model.

## Applications

NAS has been used to improve Point Cloud Understanding and Transformers, for example.

Another application is neural network-based image editing. Anycost GAN is a network created using a once-for-all network style for image editing. Instead of using the entire model to create preview images of an image we want to edit, which is too slow for proper interactive use, we can use the smaller ideal model chosen from the once-for-all network to do fast rough prediction, then use the whole model for the finalized image output.

Another application is on-device pose estimation. We can use the hardware-aware capabilities of NAS to design a model that has much less latency than the previous state-of-the-art, allowing for reasonable real-time pose estimation. This has been applied to on device person recognition, gaze recognition, and segmentation as well.

Finally, NAS can be used to design better quantum models by optimizing for more robustness to quantum noise.
