# Lecture 03: Pruning and Sparsity (Part I)

## Note Information

| Title       | Introduction to TinyML and Efficient Deep Learning Computing                                                    |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han                                                                                                        |
| Date        | 09/15/2022                                                                                                      |
| Note Author | Andrew Feldman (abf149)                                                                                         |
| Description | Background on sparsity (benefits and challenges); motivation for pruning; basic pruning techniques & key metrics|

## Outline

- Introduce neural net pruning
- Process of pruning
- Granularity of pruning
- Pruning criteria

## Background on the cost of executing DNNs

- Increasing cost of memory to store large models

    - Multiply/accumulate is the primary compute operation in DNN
    - Main-memory access cost exceeds multiply cost by roughly 200x

    ![Memory is expensive](figures/lecture-03/abf149/dnn_hw_arch_costs.PNG)

    Slide screenshot; figure from [Horowitz, M., 2014].

- Models are getting larger
- The cost of moving models from main memory into on-chip memory hierarchy & arithmetic units is increasing

### Key DNN execution cost metrics

- Model size - number of weight parameters, which directly impacts storage size and indirectly impacts the bandwidth and latency involved in transferring the model between locations.
    - The increasing cost of memory combined with increasing model size means that storage cost is a key reason for minimizing model size
    - It is also more costly to build memories which support higher bandwidth, and as models expand to have more weights, the bandwidth required to move move chunks of weight within the processor memory hierarchy (main memory, cache, register file) increases. This too motivates minimizing model size
    - Downloading a DNN onto a mobile device is another point where model size is impactful; large model size can mean the difference between being able to download a DNN model quickly over mobile data vs taking a long time to transfer the model and/or requiring WiFi
    - Example - 10 million weights may be two large to store on a phone or conveniently transfer to a phone, especially over Cellular Data
- Model execution time
    - The scaling behavior of execution time with respect to model size is a function of the arithmetic operations comprising the neural network
    - For CONV and FC (matrix multiplication) layers, 
- model inference energy
- Example - large RL policy DNNs for protein folding, gameplay, etc. may consume thousands of dollars of GPU or TPU time


## Background on sparsity

Sparsity generally referes to underlying statistical redundancy in a data structure which contains multiple values. Generally sparsity implies that there is an opportunity to *compress* the data structure by omitting redundancy. 

Owing to the desire to reduce the key cost metrics (mod)

In this lecture we primarily focus on exploiting the most trivial form of sparsity (think: redundancy), namely the subset of zero-values in the data structure.

In this class the data structures we care about are tensors and we are interesting 

## What is pruning?

Pruning - metaphorically, brain damage”; or put another way, eliminating unnecessary knowledge. A DNN's knowledge is understood to reside in its *weights*.

It turns out that many if not most industrially-interesting DNN models can be decomposed into their component weight tensors, and the component weight tensors frequently contain statistical redundancy (sparsity) - so you should be able to compress the component tensors of the DNN and thereby shrink the storage, data transfer and execution time/energy costs of the DNN. Think of this like putting each DNN tensor into a "zip file" which requires fewer bits than the original.

However, compressed tensors still need to be used for computation at inference-time, so we need a compression scheme that lends itself to efficient computation; we do not want the compression savings to be overshadowed by the cost of decompressing the tensor for computation.

Many compression techniques exist which work by removing redundant zero-values from a tensor, such that  most or all of the remaining values after compression are exclusively non-zero. Frequently it is possible to compute directly on compressed data using such a zero-removal scheme, because the computations we are interested in performing in DNNs are multiply/accumulate, for which zero operands are ineffectual.

However, in many industrially-interesting DNNs, the sparsity does not directly take the form of redundant zero-values, but rather manifests as other hidden patterns in the weight tensors. This leads to the idea of *pruning* - assuming that the tensor has some underlying statistical sparsity, we use some technique to convert this sparsity to zero-sparsity. We flip some of the DNN weights to zero and potentially adjust the values of the other weights to compensate. Since zero-valued weights are ineffectual for multiply/accumulate, we treat the zero-valued weights as if they do not exist (they were ``pruned'') and therefore we do not store, transfer or compute on these weights. So in summary, we achieve savings on key metrics by converting underlying statistical sparsity to a specific form of sparsity (zero-sparsity) and then rely on hardware and/or software to *exploit* this sparsity to do less work.

## Introducing pruning through a neurobiological analogy

In neurobiology axonal connections are understood to play a role in representing and computing on information in the brain. The number of axonal connections ramps up after birth, stays high for a period of time, until eventually a very large number of connections are destroyed toward the end of adolescence and into adulthood. Scientists assume that the brain does this in order to facilitate aggressive learning early in life, followed by honing in on only the most critical knowledge later in life. This latter stage of destroying axonal connections is referred to in neurobiology as ``pruning'' of the neuronal axons. 

The process of surfacing tensor sparsity as redundant zero weights and then avoiding doing work on these zero weights was named ``pruning'' by analogy to the latter stage of axonal destruction in human development, since the zero weights have effectively been removed from consideration, and any remaining DNN knowledge must be encapsulated in the smaller number of remaining nonzero weights. 

Sometimes, DNNs are trained to completion (pre-trained) before pruning, in which case the neurobiological analogy applies even more closely - the initial pre-training of the model is like the infant and childhood stage of aggressive learning, after which pruning is needed to preserve only the most important knowledge for "adulthood".


## Pruning process
- Select specific weights or groups of weights to prune
- Weights become zero
- Zero weights may be skipped entirely from computation ⇒ time ($), energy savings
- Zero weights do not need to be stored ⇒ storage & data transfer savings
- Neurons for which which all ingress weights are zero can itself be pruned ⇒ prune all egress weights from that neuron ⇒ additional execution & storage savings

## Specific DNN elements to target for pruning
- Weight pruning - as described in the previous section
- Neuron pruning - prune neurons with the lowest L1 norm of weights (?)
    - All ingress & egress weights to neuron may be limited
    - How is this different from channel pruning(?)
    - Lower recovered accuracy but likely more readily exploitable on CPU and GPU

![Neural Network Pruning](figures/lecture-03/abf149/pruning_obd.PNG)

Slide screenshot; figure from *Optimal Brain Damage* [LeCun et. al., NeurIPS 1989] and *Learning Both Weights and Connections for Efficient Neural Network* [Han et. al., NeurIPS 2015].

### Weight updates
- “Brain damage” of pruning → lower accuracy after pruning (”recovered accuracy”)
- Recovered accuracy can be increased to updating the weights to adapt to the brain damage

### How to choose which weights to prune?
- Use the Taylor series of network loss wrt. weight value as a proxy for the “saliency” of a weight or group of weights to the DNN’s “knowledge”
- Different Taylor series terms provide diffect information

### Optimal Brain Damage (OBD) (Yan LeCun) - early second-order information pruning
- Assumes pre-trained network ⇒ first-order term is ~0, second-order term is a function of Hessian of loss wrt weights
- Pros: effectively a “one-shot” computation of which weights to prune & how to adapt the remaining weights to correct for pruning
- Cons:
    - Generally, second-order information based on the Hessian size scales as (# weights)^2 ⇒ unweildy to compute & store
    - OBD assumes diagonal Hessian i.e. only self-second-derivatives are nonzero, however the subsequent Optimal Brain Surgeon paper (OBS) showed empirically that this approximation is frequently inaccurate, which has a negative impact on recovered accuracy
    - Also, “one-shot” computation of pruning outcome with Hessian seeks the nearest solution to the original trained local minimum; unlikely to escape local minima

### Pruning with first-order information (first-order Taylor series term) - magnitude pruning
- Developed to a large extend through Han et. al. work
- Use weight magnitude as a proxy for saliency; prune lowest-saliency weights
- Use gradient descent/SGD to compute corrective weight updates after pruning
- Very popular technique

## Pruning granularity
- Fine-grained (a.k.a. unstructured) pruning
    - High recovered accuracy
    - Potential to exploit in custom hardware
    - Challenging or impossible to exploit effectively on CPU GPU
- N:M block pruning
    - Medium recovered accuracy
    - 2:4 block sparsity exploited by NVIDIA Ampere GPU architecture
    - Likely challenging to exploit effectively on CPU
- Channel pruning
    - Lowest recovered accuracy
    - Based on L1/L2 norm of channel weight magnitude (?)
    - Feasible to exploit on CPU and GPU (effectively a change in layer geometry)
    - Also possible to exploit in custom hardware