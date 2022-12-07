# Lecture 14: Distributed Training and Gradient Compression (Part II)

## Note Information

| Title       | Introduction to TinyML and Efficient Deep Learning Computing                        |
| ----------- | ----------------------------------------------------------------------------------- |
| Lecturer    | Song Han                                                                            |
| Date        | 10/27/2022                                                                          |
| Note Author | Alex Gu (gua)                                                                       |
| Description | Bottlenecks in distributed training, gradient compression, delayed gradient updates |

## Outline of this lecture

- Understand the bandwidth and latency bottleneck of distributed training
- Overcome the bandwidth bottleneck using gradient compression
- Overcome the latency bottleneck using delayed gradient update

# Section 1: Bottlenecks in Distributed Training

First, distributed training **requires synchronization, causing a high communication frequency**. The local gradients must be synchronized and aggregated across all nodes.

![](figures/lecture-14/gua/1.png)

Second, larger models lead to **larger transfer data size, leading to longer transfer times**.

![](figures/lecture-14/gua/2.png)

Third, with more training nodes, there will be **more communication steps** and **longer latency**
![](figures/lecture-14/gua/3.png)

Finally, with a cellular network, there may be **poor network bandwidth and intermittent connection.**

# Section 2: Gradient Compression

There are two general ways to reduce gradient size, gradient pruning and gradient quantization. Let's look at some techniques for both:

## Gradient Pruning

### Sparse Communication [[Fikri Aji et al., 2017]](https://aclanthology.org/D17-1045.pdf)

Since communicating all the gradients may be expensive, one optimization is to only send the gradients with top-k magnitude

![](figures/lecture-14/gua/4.png)

As one can see, this improves training speed, but fails on modern models like Res-Net.

![](figures/lecture-14/gua/5.png) ![](figures/lecture-14/gua/6.png)

### Deep Gradient Compression [[Lin et al., 2018]](https://arxiv.org/pdf/1712.01887.pdf)

The reason this failure occurs is because when momentum updates are used, there is significant deviation between the true updates and the updates when gradients are dropped. The effect can be seen in the image below: when sparse communication is used, the pruned gradients are accumulated, and when you use accumulated gradients directly, you'll get an update in the wrong direction. The fix to this is to accumulate the momentum terms, not the gradients.
![](figures/lecture-14/gua/7.png)

In addition, in the first few epochs of changing, the weights of a neural network change rapidly, and tricks like local gradient accumulation will lead to inexact gradients, making the problem worse. Therefore, the authors of Deep Gradient Compression both warm up the learning rate and warm up the sparsity (shown below). This helps the optimizer gradually adapt to larger sparsities.

![](figures/lecture-14/gua/8.png)

The authors show that at 99.9% sparsity, the model with Deep Gradient Compression is able to perform better than various ablations.
![](figures/lecture-14/gua/9.png)

![](figures/lecture-14/gua/10.png)

### PowerSGD [[Vogels et al., 2019]](https://arxiv.org/pdf/1905.13727.pdf)

One problem with DGC is that sparse tensors become more dense after all-reduce operations:
![](figures/lecture-14/gua/11.png)

Therefore, instead of fine-grained pruning, we use a low-rank factorization of the matrix instead.

![](figures/lecture-14/gua/12.png)

We see that there is a near-linear speedup in terms of the number of workers when using PowerSGD.
![](figures/lecture-14/gua/13.png)

Nevertheless, the accuracy is kept at the same level
![](figures/lecture-14/gua/14.png)

## Gradient Quantization

### 1-Bit SGD [[Seide et al., 2014]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/IS140694.pdf)

In 1-bit SGD quantization, we use a column-wise scaling factor and use one-bit for each quantized gradient depending on whether it is positive or negative. The quantization error is then stored locally for gradient updates.

![](figures/lecture-14/gua/15.png)

![](figures/lecture-14/gua/16.png)

### Threshold Quantization [[Strom et al., 2015]](https://assets.amazon.science/57/cf/1fc5a69d4a6dbc860bd4f3e0dd64/scalable-distributed-dnn-training-using-commodity-gpu-cloud-computing.pdf)

In threshold quantization, we pick a threshold $\tau$. We quantize values with absolute value over $\tau$ to $\pm \tau$, and the rest of the values to $0$. The choice of $\tau$ must be done empirically.
![](figures/lecture-14/gua/17.png)

### TernGrad [[Alistarh et al., 2016]](https://arxiv.org/pdf/1610.02132.pdf)

In TernGrad, we quantize $\frac{g_i}{\max(g)}$ to $0, 1, -1$ with probability $\frac{|g_i|}{\max(g)}$. This way, we have that $\mathbb{E}[\text{Quantize}(g_i)] = g_i$. This ensures there is no quantization error accumulated.
![](figures/lecture-14/gua/18.png)

# Section 3: Delayed Gradient Update [[Zhu et al., 2021]](https://proceedings.neurips.cc/paper/2021/file/fc03d48253286a798f5116ec00e99b2b-Paper.pdf)

While the methods in Section 2 primarily focus on addressing the bandwidth bottleneck, delayed gradient updates help address the **latency** bottleneck. Right now, the problem with conventional methods are that local updates and communication are performed sequentially, so that the worker will have to wait for the communication to finish before the next step, as shown here.

![](figures/lecture-14/gua/19.png)
![](figures/lecture-14/gua/20.png)

The idea is delayed gradient averaging: with no delay, each local machine is blocked when waiting for synchronization. However, with a delay, the workers will continue to perform computations locally while parameters are in transmission. This way, even if the latency increases, the total training time may remain unaffected.
![](figures/lecture-14/gua/21.png)

One issue that may arise is staleness: in this example, the 3rd iteration's gradients only arrive at the 6th iteration:
![](figures/lecture-14/gua/22.png)
![](https://i.imgur.com/7xCxJ21.png)

If we directly update the weights by $w_{(i, j)} = w_{(i, j)} - \eta \overline{\nabla w_{(i-D)}}$,the model's performance will hurt because the gradients would be stale. Instead, applying gradients with correction terms, $w_{(i, j)} = w_{(i, j)} - \eta( \nabla w_{(i, j)} - \nabla w_{(i-D, j)} + \overline{\nabla w_{(i-D)}})$, mitigates this issue, as seen here:
![](figures/lecture-14/gua/23.png)

We can see that delayed gradient averaging works well on real world benchmarks as well!
![](figures/lecture-14/gua/24.png)
