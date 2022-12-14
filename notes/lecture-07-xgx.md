# Lecture 07: Neural Architecture Search (Part I)

## Note Information

| Title | Neural Architecture Search (Part I) |
| --- | --- |
| Lecturer | Song Han |
| Date | 09/29/2022 |
| Note Author | Guangxuan Xiao (xgx) |
| Description | Basic concepts of NN archs, design principles of sever typical mannulally desing neural network arhcs. Introduce nn search, autnomatic tech for desinging nn archs. |

## Basic concept and design principles of NN

### Recap: Primitive Operations

- Linear Layer: c_o c_i
  
- Convolution layer: c_o c_i k_h k_w h_o w_o
  
- Grouped Convolution Layer: c_o c_i k_h k_w h_o w_o / g
  
- Depth-wise Convolution Layer: c_o k_h k_w h_o w_o
  
- 1x1 Convolution: c_o c_i h_o w_o
  

### Basic Concepts

#### Stage

Input stem, the head, and several stages.

Early stages have larger feature map sizes, so we need to keep the width small to reduce the cost. Late stages have smaller feature map sizes so we can increase the width.

#### Downsample

- feature map down sampling is usually done at the the first block in each stage via stride convolution or pooling

#### Residual / Skip connection

We can residual skip connections for the remaining blocks as their input.

## Manually-designed Neural Network

- AlexNet: large-kernel conv in early stages
  
- VGG: stacking multiple 3x3 convolution layers
  
- SqueezeNet: Replace 3x3 conv with fire module
  
  - Fire module: reduce the cost by using 1x1 connvolution
    
  - reduce the cost of 3x3 connv layers
    
- ResNet50: bottleneck block
  
  - Feed reduced feature map by 3x3
    
  - Expand channels by one by one
    
  - Reduced by 8.5x
    
- ResNeXt: grouped convolution
  
  - Replace 3x3 conv with 3x3 grouped convolution
    
  - Equivvalent to a multi-path block
    
  - Smaller but more channels
    
- MobileNet: depthwise-separable block
  
  - Depth wise connv is an extrem case of group convolution
- MobileNetV2: inverted bottlenect block
  
  - Depthwise conv has a much lower capacity compared to normal conv
    
  - Increase the depth wise conv input and output channels to improve its capacity
    
  - Depthwise convolution's cost only grows linearly. Therefore, the cost is still affordable.
    
- ShuffleNet: 1x1 group convo & channel shuffle
  
  - Further reduce the cost by replacing 1x1 convulution with 1x1 group convolution
    
  - Exchange information across different groups via channel shuffle.
    
  
  ##
  
  ## NAS: From Manual Desing to Automatic Design
  

### Illustration of NAS

Components and goal

find the best nn arch in the search space, maximizing the objective of interest (e.g., accuracy, efficiency, etc)

Search space -> search strategy <-> Performance estimation strategy

Search space: example

- Search space is a set of candidate of nn archs
  
- Search strategy defines how to explore the search space
  
- performance estimation: what is a good model?
  

### Cell-level search space

Classifier have normal (stride = 1) and reduction (stride > 1) cells.

NASNet cell-level search space

- Left: RNN controler generates the candidate cells five steps: finding two
  
- Right: a cell generated after one step
  

### Network-level Search Space

#### Fixed topology

#### Searchable topology

NAS-FPN

Design the search space for TinyML

### Search Strategy

#### Grid search

compound scaling on depth, width, and resolution

#### Random search

single-path-one-shot, with a good search space

#### Reinforcement learning

Challenge: R (accuracy) is a non-differentiable objective. How to update the RNN parameters?

Binarization process is non-differentiable, and it is possible to use RL.

#### Bayesian optimization

- Idea: balacing exploitation and exploration with the acquisition function (i.e.. waht is the next arch to search)

#### Gradient-based search

#### Evolutionary search

mutation + cross-over
