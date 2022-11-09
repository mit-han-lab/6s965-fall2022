**Lecture 15: On-Device Learning and Transfer Learning**
Part I

***Motivation***
- Customizations: AI systems want adapt from new data collected from edge device sensors.
- With cloud-based computing, the new data is sent to the Cloud Severe where it learns. Then the nenw training data is sent back to the edge device.
- However, some sensitive data is collected and it must not leave the Edge device to train on a cloud server because of security and regulations.
- For this reason, on-device learning offers the users: better provacy, lwoer cost, customization, and life-long learning.

*IoT Devices*
- It's a good idea to squeeze training into IoT Dveices, since there are billiions of them aroundd the world (based on microcontrollers)
- *Low-cost*: they are cheap and accessiible, even low-income people can afford to access. This would aid in Democratizing AI.
- *Low-power*: they don't require as much energy as servers. If reused or repurposed, they are green AI which reduces carbon footprint.

***Efficient On-Device Learning Algorithms***

*Understand the training bottleneck and improve training efficiency*
- A challenge of on-device training is that the memory of the edge devices is usually too constrained to hold DNNs compared to the cloud AI's capacity (up to 100x smaller).
- In order to reduce the size of the models we want to run in these tiny memories, we need to reduce both the weights.

**Memory Bottlenecks**

Training is the memory bottleneck
- Why is the training memory much larger than inference (eg. 8x)?
    - This is because of the intermediate activations!
- Inference does not need to store activations, but training does.
- The activations then grow linearly with batch size (batch size = 1 always for inference).
- Even when the batch size is jsut one, the activations are usually larger than the model weights.

Activation is the memory bottleneck
- Parameters do not bottleneck, they are only a small fraciton (eg. 7x) of the memory
- Previous methods seen focus on reducing the number of parameters or FLOPS, but because they don't focus on activation the main bottleneck does not reduce by much.

Parameter-Efficient Transfer Learning
- *Full*: By finetuning the full netwroks, we observe better accuracy but remain very inneficient.
- *Last*: By finetuning the last classifier head we are now more efficient but the memooory capacity remains limited.
- *BN+Last*: By finetunning the BN (batch normalization) layers as well as the last layers, then we become parameter efficient but the memory saviing is limited (not as high as we'd like). Addditionally, we observe a significant loss in accuracy.
- *TinyTL*: (Memory-efficient transfer learning) Only finetune the bias AND lite residual learning. Yields high accuracy and large memory savings.

TinyTL
- Updating weights is memory expensive because it requires storing intermediate activationis, whereas updating the bias is memory efficient
- By freezing the weights and only finetuning the biases we can save up 12x the memory. This on its own, hurts the accuracy.
- To maintain the accuracy,  we add lite residual moddules to increase model capacity.
- **Key Principle:** *Keep the activation size small*
    1. Reduce the resolution
    2. Avoid inverted bottleneck
- For 1/6 channel, 1/2 resolution, and 2/3 depth, there is a 4% activation size.
- Provides up to 6.5x memory saving withoutt acccuracy loss.
- Specialized models for different tasks: The relative accuracy order between different pre-trained models changes significatnly among ImageNet and transfer learning datasets. This motivates personalized and specialized NN architecture for different downstream tasks.
- In comaprison with dynamic activation pruning, TinyTL saves memory more efficiently.
- In-memory training
    - TinyTL supports batch 1 training by group normalization.
    - In conjunction with lite  residual mode, it reduces the cost o ftraining memory to 16MB (fits in L3 cache).
    - This enables using training process into cache, which is more energy efficient than training on DRAM.

**On-Device Learning Principles**
- Parameter-efficiency does not translate to memory efficiency
- The main memory bottleneck is not the parameters, but the training activations.
- TinyTL saves the activation memory by compounding finetuning the bias only with lite residual learning.

***Co-Designs for On-Device Training***

*The difficulty of optimizing quantized model and system-level support*

- *Training* is more expensive than *inference* because of back-propagation (makes it hard to fit IoT devices).

On-Device Training Under 256KB Memory
1. Quantization Aware Scaling
2. Sparse Layer/Tensor Update
3. Tiny Training Engine

**1. Quantization Aware Scaling**
- *Challenges* : When running quantization aware training (using fake quantization graphs), most of the intermediate tensors are still inn FP32 format, which does not save any memory.
    - However, using real graphs to quantize (int8/32) does save the memory space but it's really hard to quantize because of:
        - mixed precisions (int8/32/fp32)
        - no batch normalization
    - Also, the training convergennce is worse because the scale of the weightt and gradiennts do noot match the real quantized training.
- *QAS* : Quantiztaion-Aware Scaling
    - Addresses the optimization difficulty of the quantized graphs
    - It works by aligning the weights-to-gradient ratio with the fps32's weights.
    - When we compare the perfromance, we see that tthe accuracy is closer (if not better) than when thte model uses fp32, using way less memory.

**2. Sparse Layer/Tensor Update**
![](notes/figures/lecture-15/solr/updates.png)
- *Full Update* : Updating the whole model's biases and weights is too expensive.
    - We need to save the activation (which is large, and the memory bottleneck)
    - We also need to store the updated weights in SRAM (because the flash is read-only)
- *Last Layer Update* : We can update only the last layer, since it is cheaper.
    - There is no need to do back propagation to the previous layers.
    - However, the acccuracy is lower and that is not ideal.
- *Bias-Only Update* : Updating only the biases is cheaper.
    - We don't need to store the activations (saving memory).
    - However, we need to back propagate all the way back to the first layer.
- *Sparse Tensor Update* : Updating only sparse tensors/layers. 
    - There's no need to back propagate to the first layers.
    - We only need to store a smaller subset of the activations.
    - This method reudces the amount that needs to be updated.
- How do we know which layers to update?
    - Starting layers have really high activation ccost
    - The later layers have a high weight cost (because they contain more info). In this layer we only update the bias (related to activation only).
    - The middle layers have a low overall memory cost. For these layers we only update the weights (related to activation and weights).
    - When we update the bias, the accuracy increases as there are more updated layers. Hoowever, it plateaus or saturates really early. 
    - When we update the weights, the later layers are more important. *The first point-wise convolution contributes more.*

**3. Tiny Training Engine**
![TTE](notes/figures/lecture-15/solr/tte.png)
- Previoous deep layer training frameworks focus on flexibility and the auto-ddiff is performed at runtime. This means that we cannot apply many graph optimizations during runtime because they will leadd to runtime overheadd. 
- Existing frameworks have trouble training into tiny devices because:
    - Runtime is heavy
        - There are heavy dependencies and a large binary size (more than 1000MB of static memory).
        - Autodifff only happens at runtime
        - Operators optimized for the cloud, not for edge devices
    - Memory is heavy
        - There's a lot of intermediate and unused buffers.
        - They have to compute the ffull gradients.
- Tiny Training Engine (TTE)
    - TTE moves the worrkload from runtime to compile.time, which minimizes the runtime overhead.
    - This also allows for extensive graph optimmizations.
        -Some of the graph level optimizations are:
            - Sparse layer/tensor updates
            - Operator reordering and in-place update
            - Constant folding
            - Dead-code elimination

- Sparse layer/tensor updates
    - In bias-only updates we note whether the tensor requires a gradient or not, then we remove unnecessary computations using code elimination (from DAG) via a dependency analysis.
    - We can ffreely annotate any paramters as TTE will trim the computation accordingly.

- Operator reordering and in-place update
    - The operatioon life-cycle analysis reveals the memroy reddundancy in the ooptimizatioon step. This wastes our memory.
    - First we calculate all the gradients and then apply one-by-one.
    - We note that hthe intermediate buffers consume a lot of spaces.
    - The gradient updates are then immediately applied once they are calculated and then can be released.
    - With reordering, we can update gradients immediately, which can be released even earlier before back-propagating to earlier layers
        - This leads to peak memory reduction and model speedup.

- TTE provides a systematic support for sparse update schemes for vision and NLP models, which leads to consistenn memorry saving at the same training accuracy.

***On-Device Training Take-Home***
- Quantized Training
    - Fake quantization does not save memory
    - Real quantiization does, but is hard to optimize. 
    - Quantization Aware Scaling (QAS) can optimized a quantized graph.
- Sparse Update
    - Sparse learning happens in the human brain
    - We only update important layers and parameters to save memory.
- System Support
    - We move workloads to compile.time (eg. auto-diff) to minimize the runtime cost.
    - Optimized schedules and kernels improve throughput.

***Summary***
- On-device transfer learning algorithms
    - Training memory bottleneck stems from the activation
    - Efficient transfer elarning can be done with bias-only and lite-residual
- The co-design points of on.devicce training
    - Why is training a quantized model is difficult? How to improve?
    - Doing a full-update is too expensive; using a sparse update for on-device training.
    -System support for efficient on-device training