# Lecture 09:  Neural Architecture Search (Part III)

## Note Information

| Title       | Neural Architecture Search (Part III)                                                                                           |
|-------------|-----------------------------------------------------------------------------------------------------------------|
| Lecturer    | Song Han \& Lucas Liebenwein                                                                                                       |
| Date        | 10/06/2022                                                                                                      |
| Note Author | Tara Boroushaki (tarab)                                                                                      |
| Description | In this lecture, Omnimizer, a tool for neural network hardware-aware optimization, was introduced. 
 

## Why do we need hardware-aware algorithm optimization?
- Accelerate design and developement of neural networks
- Give developers control through self-service tools
- Faster Model inference 
- Getting the best of current available hardware 


### Edge AI applications:
- Autonomous driving and Advanced driver assistance systems 
- Robotics
- Internet of Things (IoT)
- Augmented Reality and Virtual Reality Headsets

### What is the problem that Omnimizer is trying to solve?
There is a big gap between training neural networks and hardware deployment. It takes a long time to deploy AI models on actual hardwares. Even after deployment, there is a significant gap between expected performance and how the AI performs on the hardware. With current setups, a huge R&D cost and investment is always needed.


## Omnimizer workflow:

#### Current workflow:

The ML engineers develop and train a network, providing the model to the deployment engineers who implement the network on the intended hardware. The ML engineers have to wait to get feedback from the deployment engineer on the latency, memory, and energy consumption and change the network to match the requirement of the intended product.
This process is very time consuming.

#### How Omnimizer changes the overall workflow:
Omnimizer provides the ML engineer with a platform to deploy the network on an actual hardware and get the latency, memory, and energy consumption estimation. This capability shorten the R\&D cost and investment and reduces the time needed to build new products.

Omnimizer can be added to Pytorch and enables the ML engineer to deploy and test on target hardware through Omnimizer Core and Engine.

## What are other approaches and competitors?

### Microsoft (Neural Network Intelligence): 
- Has a collection of algorithm
- Needs manual search space design 
- Works good only on well-known benchmarks such as ImageNet

### Google (Vertex AI):
- A black box design
- Works well only on established benchmarks such as ImageNet

### Omnimizer: 
- Automatic optimization and adaptation
- Can work with any training environment 
- Instant deployment on hardware (that are supported)


## Omnimizer Workflow:
### Setup:
OmniML has several actual hardwares that their server deploys and measures latency.
```
from omnimizer import engine
deployment = {
    "device": "S888",
    "precision": "int8",
}
engine.get_latency(model, deployment)
```

### Diagnosis & Adaptation:
The adaptation methods are based on real hardware deployments.
```
adapted_model = nas.adapt(model, deployment) 
engine.profile(adapted_model, deployment)
 
```

### Train and Optimize:
Uses FastNAS (Faster but smaller search space) or AutoNAS (Slower but larger search space).
```
from omnimizer import nas
omni_model = nas.omnimize(model)
train(omni_model, dataloader)
```

### Search:
Searches for the subnet that meets the required latency restrictions and has the optimal accuracy. 

```
from omnimizer import nas
constraints = {
    "latency": 3.0,  # ms
}
# adaptive to "autonas" and "fastnas?
sub_model = nas.search(
    model,
    constraints,
    deployment,
)
 ```

 ### Deploy:

```
from omnimizer import engine 
device_model = engine.compile(sub_model)
device_model.get_latency()
out = device_model(sample_input)
 ```


 # Conclusion:
 
**Omnimizer NAS**:
 - Pytorch-native model optimization and adaptation
 - Can do both AutoNAS and FastNAS
 - Searches for optimal subnet

**Omnimizer Engine**:
 - Cloud-native interface for deployment on real hardware
 - Accurately & simple model profiling on real hardware
 - on-device inference
 