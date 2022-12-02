# Lecture 16: On-Device Training and Transfer Learning (Part II)

## Note Information

| Title       | Lecture 16: On-Device Training and Transfer Learning (Part II)                                                  |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 11/04/2022                                                                                                      |
| Note Author | Pranav Krishna (pkrishna)                                                                                       |
| Description | The second half of a two-lecture series on On-Device Training and Transfer Learning                             |


## The Tiny Training Engine (TTE)

There are two major reasons why existing frameworks cannot work for training on tiny devices; this is because:
1. **Runtime** is heavy: Heavy dependencies and large binary size (>90 MB), the Autodiff step is done at runtime, and in general operators are optimized for the cloud setting (large memory) and not for tiny applications.
2. **Memory** is heavy: There are a lot of intermediate (and sometimes unusued) buffers, and no support for sparse backpropagation.

To fix these problems, TTE was proposed. There are a few changes to the original system that TTE makes:

### Moving Computations to Compile-Time

The first major step that TTE makes is to move the Autodiff procedure to Compile-time rather than runtime. This reduces the runtime overhead, and allows for the system to also make graph optimizations, which can also be done in compile time.

### Graph Optimizations

There are a variety of methods that this can be done. The three most common methods are:
1. **Bias-only Updates** - as implied, only the bias is updated; in compile time, steps used to update weights are done first
2. **Sparse Layer Updates** - in this scheme, only select layers are updated each step,
3. **Sparse Tensor Updates** - in this scheme, only select parts of each weight tensor are updated

### Operation Reordering

Another quirk about the general scheme for cloud-based models is that all the gradients are computed, and only then are all the weights updated. This is not, as you may imagine, efficient for memory; so, TTE reorders the operations such that once the gradient is calculated, the weight layer is updated. These updates can be done in-place for a further boost in memory saved.

### Extending TTE to Other Platforms

The authors of the paper that developed TTE also extended it to other 'tiny' processors.

## Privacy Leakage in Federated Learning

### Federated Learning

*Federated Learning* is an umbrella term given to a class of strategies that solve the problem of training a model across data that are stored in a variety of decentralized servers, with the requirement that this data cannot be transferred across servers. Most commonly, the reason that this data cannot be transferred in the real-world is due to privacy concerns.

One common approach is the FedAvg algorithm - given a baseline model, send this model to all the servers, train (update for $N$ iterations) on the data in parallel on each of these servers; then, the center averages all the weights of the resultant models and sends them out. This is repeated until some predetermined termination condition is reached. But is this truly safe?

### Safety Concerns

We focus on a different, but related question to what ended the previous section - can we reconstruct the input data given the gradients? There are a number of works on the topic:
1. *Membership Inference:* Given the gradients, it is possible to determine whether a datapoint was in the batch or not.
2. *Property Inference:* Given the gradients, it is possible to determine whether or not there exists a datapoint with a certain property in the batch.
3. We can reconstruct the input pretty well!

### Deep Leakage by Gradient Matching

The general strategy is as follows: initialize a dummy input, label pair; compute the gradients of the model with respect to the dummy input, and use a distance function as the loss to update the dummy input. Not only does this work, but this method is also resistant to Gaussian and Laplacian noise, unless you want to sacrifice a sufficient amount of accuracy. In addition, Half-Precision Quantization (i.e. 16 bits) also cannot protect against this type of attack. Gradient compression can work, but a ratio >30 % is needed to be resistant.

