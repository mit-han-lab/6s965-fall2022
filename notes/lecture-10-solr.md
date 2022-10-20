Lecture 10: Knowledge Distillation

-   Widely used for NNs to fit in tiny devices

-   Lecture Outline:

1.  What is knowledge distillation

1.  What are student vs teacher models?

2.  How to transfer knowledge from teacher to student model?

3.  What to match?

1.  What are the things we want to learn about the teacher?

5.  Self and online distillation

6.  Distillation for different tasks

7.  Network Augmentation, a training technique for tiny machine learning models.

What is knowledge distillation

-   CloudML has huge computation resources (19.6T FLOPS) and sufficient memory (80GB)

-   Can accommodate large NN like ResNet, VIT-Large, etc. 

-   Tiny AI has  only mini FLOPs per second and limited storage (256kB)

-   Smaller models: MCUNetand MobileNetV2-Tiny

-   Hardware is a challenge for TinyML, since neural networks must be tiny to run and be efficient on tiny edge devices.

-   Tiny models are hard to train because they underfit large datasets 

-   Accuracy drops with smaller models because we have smaller capacity and  resources

Distilling the Knowledge in a Neural Network [Hinton et al. NNeurIPS Workshops 2014]
- (See Fig. 1)

-   Simple but effective method for transferring knowledge from teacher to student model

-   Teacher Model: Larger trained model

-   Student Model: Smaller trained model

-   Basic idea illustration of a knowledge distribution ->

-   New loss: Distillation Loss

-   Distillation Loss: Loss from trying to match the logits in student model from teacher model

Matching Prediction Probabilities between Teacher and Student

-   See Fig. 2

-   Take in a prediction image (cat) and pass it through teacher and student model 

-   We compare the probabilities of classification for both. We see that the logits in the student model show that it's less confident of it.

-   That's why it's helpful to match both the probabilities and the logits (our motivation for distillation).

-   Temperature: We divide by the temperature too further smooth the probabilities

-   When we are trying to match the logits, we apply the temperature too improve our prediction

+   Formal Definition:

-   We use a softmax function to generate the logits as Zi to probabilities PZi, T (where T is the temperature). See Fig. 3

-   Here, i,j = 0,1,2,...C-1, where C is the number of Classes.

-   Goal: The goal of using knowledge distillation is to align the teacher and student's class probability distributions

What to match

1.  Output Logits
-   See Fig. 4
+  As before, matching the logits helps the student model become more confident in its prediction

+  We use cross-entropy and L2 losses as out distillation losses

2.  Intermediate Weights
- See Fig. 5

+  Rather than matching the final prediction we can try to match weights in the middle of the model.

+  We calculate the distillation loss from the in-between layers (rather than the logits)

+  We also add L2 loss to calculate the distance between the weights and thus, its loss.

+  Using a fully-connected layer, a linear transformation is applied to match the student model's dimensions to the teacher model's.

3.  Intermediate Features

+  Think of a NN as an automatic feature extractor, so we try to match the features between the teacher and student models

+  We minimize maximum mean discrepancy (MMD) between the feature maps

+  Intuitively, both models should have similar feature distributions, not just output probability distributions

*  We use MMD as an objective where k is the kernel function (dot product)

*  Think of it as calculating the cosine distance between the teacher and student's feature map.

+  The close they are, the larger the term is (so the smaller the loss, because it's negative). See Fig. 6

1.  Minimizing L2 distance between feature maps

1.  We can also use the teacher's feature map from m to m x k dimensions and then back to m.

1.  K is a factor, and usually given the value 0.5

3.  This paraphrasing of features is supervised by a reconstruction loss, which compares against the original m dimension output.

4.  The student uses one layer of MLP to obtain a factor close to the m x k dimensions

5.  The FT loss minimizes the distance between the factors from the student and teacher (See Fig. 7)

1.  Gradients

+  Gradients of feature maps are used to characterize the "attention" of DNNs

2.  Attention: the dL/dx, where  L is the learning objective

1.  Intuitively, if the attention is large at i,j, a small deviation to the value of at i,j will greatly impact the output.

2.  The network will pay more attention at these locations.

3.  Conversely, if the attention is lost, the value can be greatly perturbed without much consequence on the output.

4.  Different inputs and models (student vs teacher) will produce different attention maps.

4.  To match the attention maps, we use the attention transfer objective see Fig. 8

1.  Here, JS is the student's attention map and JT is the teacher's and𝛽 is a constant.

1.  Attention maps of  well-performing models like ImageNet and ResNets are similar

1.  Sparsity Patterns

1.  Intuitively, both models should have similar sparsity patterns after the  ReLu activation

1.  A layer is activate after ReLu if its value > 0 (See Fig. 9)

2.  We denote this with an indicator function:

3.  Our goal is to minimize the absolute difference between the sparsity between the student and t teacher models (See Fig. 10)

1.  It is different from matching the intermediate features because we don't care about the number of features, just whether they are activated or not.

1.  Relational Information

1.  Recall that student models differ from their teacher in the number of layers that it contains, not in the number of channels

2.  Relations between layers

1.  We can match activation functions

1.  We can match each activation layer using the inner product, result in the dot products between the teacher and the student

4.  Relations between samples

1.  We can to match the same image at each model

2.  We can also add variation between inputs that have slight variations so we can look at the intermediate features from multiple inputs

1.  First we find out the relations between samples

2.  Then we try to match the distance of the feature maps between them.

Self and Online Distillation

-   We know that the teacher model is larger than the student model and that it is fixed

-   How do we first develop a teacher model?

-   What are the advantages of large fixed teachers? Do we need one in KD?

-   To make a teacher model smaller we must get rid of computational overhead

-   Self Distillation

-   To keep the same architecture without having to train a whole new model

-   Born-again NNs: Generalizes defensive distillation by adding iterative training and using both classification and distillation objectives in the following stages (See Fig. 11)

-   Network architecture T=S1=S2=....=Sk and Network accuracy T<S1<S2<....<Sk

-   S1 is initialized randomly

-   Experimental knowledge only

-   After each generation we get one student model

-   The family of NNs can be combined for better performance

-   Regression tasks are harder to distill

-   Hard to find the probability distribution, unlike classification tasks

-   Online Distillation

-   Deep Mutual Learning (See Fig. 12)

-   Initializes two random networks

-   They can have similar or different architecture

-   There is no need to pretrain the teacher network (T=S is allowed).

-   For both teacher and student networks we want to add a distillation objective that minimizes the output distribution of the other party

-   Cross Entropy loss:
-   See Fig. 13

-   Combined

-   ONE (On-the-fly Native Ensemble): Generates multiple output probability distributions and ensembles them as the target distribution for KD.

-   Use as the teacher network

-   Requires no pre-training

-   Like Deep Mutual Learning (DML), ONE allows T=S and there's no need to pretrain.

-   Another advantage of ONE is that you don't need to train two models unlike DML

-   Be Your Own Teacher: Deep supervision and distillation

-   Uses the deeper layers to distill shallower layers, we don't have to go through all the layers

-   Intuitively the labels at the later stages will be more reliable because of the previous work, so we just have to use these to supervise the predictions from the previous layers are working fine

-   We don't need to run inference for other networks

-   Improves on accuracy but worsens in acceleration the deeper the layer

Distillation for different tasks

-   KD for object detection

-   Popular in edge devices

-   Feature Imitation

-   We try to match the intermediate activations for teacher and student model

-   We also use a 1x1 convolution to match the feature map shape

-   Besides using feature imitation, the student model is also trained to mimic the teacher model's feature maps.

-   Bounding box

-   We also use different weights for foreground and background classes (See Fig. 14)

-   We also penalize the value of the Bounded regression loss if it is greater than the loss upper bound.

-   We exploit the teacher's prediction as an upper bound for the student to achieve (See Fig. 15)

-   Once the student goes beyond that of the teacher + a margin (Student is good enough), then the loss can be 0 (no penalization)

-   This converts the bounding box regression to a classification problem)

-   Since distillation would not work on normal regression since it needs to be a real number

-   We can do this by dividing the box into a gridded doman:

-   Localization Distillation

-   Calculates the distillation loss between 2 predicted probability distributions (teacher and student)

-   KD for Semantic Segmentation

-   Want to segment what region is what (like in images: background, forests, etc.)

-   Feature imitation is similar in classification and detection

-   We add a discriminator network fo the masks which provides an adversarial loss

-   Here the student is trained "adversarially" to fool the discriminator that tells whether it's form the teacher or student

-   In the end, the student will yield high-quality region maps

-   KD for GAN

-   We combine the training for student and teacher

-   We use three losses to minimize the reconstruction, distillation, and GAN loss (See Fig. 16)

-   KD for NLP

-   We can use feature imitation to apply KD to an NLP

-   In this case, the student also tries to mimic the model's attention map

Network Augmentation

-   A training technique for tiny machine learning models.

Conventional Approach

+   Used during training to avoid overfitting

-   Data Augmentation:
+   See Fig. 17

*   We cutout or block parts of images

*   Mix up and overlap images

*   Have multiple edits of an image

-   Dropout:

+   Removes nodes to avoid overfitting certain blocks or channels

Tiny Networks

-   Network Augmentation

+   We augment the model to get extra supervision during training for tiny models

+   We build an augmented model that shares the weight with the tiny model

+   During training the data goes through forward and backward propagation, not just in the tinny model but also the large one

+   We receive two copies fo the weight (tiny and augmented model)

+   During each iteration step we use different supersets of the model

+   Increases training and validation accuracy

+   For tiny network: validation and validation acc increase

*   For large NN: training acc improves, but  hurts validation acc
