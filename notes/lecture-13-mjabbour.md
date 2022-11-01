# Lecture 13: Distributed Training and Gradient Compression (Part I)

## Note Information

| Title       | Distributed Training and Gradient Compression (Part I)                                               |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| Lecturer    | Song Han                                                                                               |
| Date        | 10/25/2022                                                                                             |
| Note Author | Mark Jabbour (mjabbour)                                                                                         |
| Description | Introduces approaches to distribute the workload of training ML models accross different machines, and the trade-offs between them. 

### Lecture overview
1. Motivation for distributed training
2. Data and Model Parallelism
3. Data parallelism in depth
4. Distributed Communication Primitives
5. Model Parallelism in depth
6. Beyond model parallelism

### 1. Motivation for distributed training


The most accurate machine learning models have become increasingly large. Making the models much slower to evaluate, and much harder to train. This has led to increased interest in effecient machine learning. While techniques like quantization and pruning help reduce the inference time, most of them are not as effective for training. Furthermore, models that contains tens of billions of parameters would not fit in a single GPU even if quantized.

For example, GPT-3 contains 175 Billion parameters. Even if they were each 8-bit quantized, this would amount to 1.7 TB, which is more than what most accelerators can fit in memory.

![Increase in model size](./figures/lecture-13/mjabbour/figure1-modelsize.png) 


Clearly, the increase in size makes training a bottle-neck for machine learning professionals. As illustrated by the following table of estimates for the training time of different models on single NVIDEA A100 GPU:


![Increase in training time](./figures/lecture-13/mjabbour/figure2-trainingtime.png) 


![Impact on the industry](./figures/lecture-13/mjabbour/figure3-meme.png)


### 2. Data and Model Parallelism

### 3. Data parallelism in depth

### 4. Distributed Communication Primitives

### 5. Model Parallelism in depth

### 6. Beyond model parallelism

