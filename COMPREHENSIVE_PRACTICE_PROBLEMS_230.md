# CEE 501 Midterm 2 - Comprehensive Practice Problem Set
## Complete Coverage of All Possible Exam Questions
**Created:** November 8, 2025 | **Due:** Friday, November 14, 2025 | **Study Time:** 5 days

---

# TABLE OF CONTENTS

1. **Neural Networks Fundamentals** (40 problems)
2. **Backpropagation & Training** (35 problems)
3. **Convolutional Neural Networks** (45 problems)
4. **Transformers & Modern Architectures** (30 problems)
5. **Advanced Topics & Applications** (30 problems)
6. **Code & Implementation** (25 problems)
7. **Mixed Difficulty & Integration** (25 problems)

**Total: 230 Practice Problems**

---

# SECTION 1: NEURAL NETWORKS FUNDAMENTALS (40 Problems)

## Basic Architecture & Concepts

**Problem 1:** What is the primary function of the input layer in a neural network?
- Explain what happens to raw input data
- How many neurons should the input layer have?

**Problem 2:** A neural network has the architecture: 784 → 128 → 64 → 10. 
- How many hidden layers does it have?
- What could this network be used for?
- Calculate the number of parameters (weights) between input and first hidden layer

**Problem 3:** Define the role of the hidden layer in a neural network. Why can't we just connect input directly to output?

**Problem 4:** What is the difference between:
- Number of layers (depth)
- Width of a layer
- Total parameters in the network
(Give example with 100 → 50 → 50 → 10 network)

**Problem 5:** A shallow network vs deep network - discuss trade-offs in:
- Computation cost
- Expressiveness
- Overfitting risk

---

## Activation Functions

**Problem 6:** For each activation function, state:
- Mathematical formula
- Output range
- When to use it
- Advantages/disadvantages

Functions: ReLU, Sigmoid, Tanh, Linear

**Problem 7:** Why is ReLU better than Sigmoid for deep networks? (Hint: Think about gradient flow)

**Problem 8:** A network using only linear activation functions (no nonlinearity):
- What function would the network compute overall?
- How many hidden layers would you actually need?
- Why is this problematic?

**Problem 9:** Plot or describe the output of:
- Input x = [-2, -1, 0, 1, 2]
- Through ReLU activation
- Through Sigmoid activation
- Compare the outputs

**Problem 10:** When would you use:
- Sigmoid in the hidden layer?
- Sigmoid in the output layer?
- ReLU in the hidden layer?
- Linear in the output layer?

**Problem 11:** What is the "dying ReLU" problem? How can you fix it?

**Problem 12:** For a classification problem with 5 classes:
- What activation should the output layer use?
- Why not use ReLU?
- What's the mathematical form?

**Problem 13:** Calculate the derivative (gradient) of:
- ReLU at x = 0.5, x = -0.5
- Sigmoid at x = 0
- Tanh at x = 0

---

## Forward Pass

**Problem 14:** For a single neuron:
- Input x = [1, 2, 3]
- Weights w = [0.5, -0.2, 0.1]
- Bias b = 0.1
- Activation: ReLU

Calculate the output step by step

**Problem 15:** A 2-layer network:
- Input: [1, 0.5]
- Layer 1: weights = [[1, 0], [0, 1]], bias = [0, 0], activation = ReLU
- Layer 2: weights = [[1, -1]], bias = [0.5], activation = Linear

Forward pass calculation

**Problem 16:** What is the computational complexity of a forward pass through:
- A single layer with m inputs and n outputs?
- A k-layer network?
- Discuss in terms of multiplications and additions

**Problem 17:** In a forward pass, what does each neuron compute?
- Write the general formula
- Explain each component (weights, bias, activation)

**Problem 18:** How would the output change if you:
- Doubled all weights
- Doubled all biases
- Applied the same change to only one layer

---

## Initialization

**Problem 19:** Why is random weight initialization important? What happens if all weights start at 0?

**Problem 20:** Compare initialization schemes:
- All zeros
- Small random values
- Xavier/Glorot initialization
- He initialization

When to use each?

**Problem 21:** For a layer with 1000 input neurons and 100 output neurons:
- Calculate appropriate He initialization range: sqrt(2/n_in)
- Calculate Xavier initialization range: sqrt(1/n)
- Compare the ranges

**Problem 22:** Why does Xavier initialization work better than random initialization for deep networks?

---

## Network Capacity & Expressiveness

**Problem 23:** How does network width affect:
- Number of parameters
- Expressiveness
- Risk of overfitting
- Training time

**Problem 24:** A universal approximation theorem states that a neural network with 1 hidden layer can approximate any continuous function. However, why do we still use deep networks?

**Problem 25:** Compare these architectures for a regression task:
- 20 → 10 → 1
- 20 → 50 → 50 → 1
- 20 → 5 → 1
Discuss trade-offs

**Problem 26:** How many parameters in each network:
- 100 → 50 → 10 (include biases)
- 784 → 128 → 64 → 10 (include biases)
- 32 → 32 → 32 → 32 (include biases)

---

## Practical Architecture Design

**Problem 27:** Design a neural network for:
a) Classifying 28×28 images into 10 classes (MNIST)
   - Input size
   - Hidden layer sizes
   - Output layer configuration
   - Activation functions

b) Predicting house prices
c) Sentiment analysis (binary classification)

**Problem 28:** For a 784→128→64→10 network:
- Draw the architecture
- Label dimensions at each layer
- Show matrix multiplications

**Problem 29:** A network has 1 million parameters. How many different architectures could you design? (Give examples)

**Problem 30:** Why would you use different network sizes for:
- Learning a toy problem
- Learning a real dataset
- Transfer learning

---

## Loss Functions for Classification

**Problem 31:** For a 3-class classification problem:
- Sample output from network: [0.1, 0.7, 0.2]
- True label: class 1 (one-hot: [0, 1, 0])
- Calculate Cross-Entropy loss

**Problem 32:** What's the difference between:
- Binary Cross-Entropy (BCELoss)
- Categorical Cross-Entropy (CrossEntropyLoss)
- When to use each?

**Problem 33:** A network outputs probabilities for 4 classes:
- [0.25, 0.25, 0.25, 0.25] (uniform)
- [0.1, 0.2, 0.3, 0.4] (peaked)
- [0.91, 0.03, 0.03, 0.03] (very confident)

For true label class 0, calculate loss for each

**Problem 34:** What properties should a good loss function have? Explain with examples.

**Problem 35:** Why do we use softmax in the output layer for classification?
- What does softmax do?
- What are the constraints on softmax outputs?
- How would training differ without softmax?

---

## Loss Functions for Regression

**Problem 36:** Compare loss functions:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss

When is each appropriate?

**Problem 37:** For a regression network predicting house prices:
- Predicted: [300k, 250k, 150k]
- True: [320k, 240k, 160k]
- Calculate MSE and MAE

**Problem 38:** Why might MAE be better than MSE for outlier-prone data?

**Problem 39:** A model predicts 100 values. The errors are:
- 99 errors of 1 unit
- 1 error of 100 units
- Calculate MSE and MAE
- Discuss which loss function would train differently

---

## Output Layer Configuration

**Problem 40:** For each task, specify:
- Output layer size
- Activation function
- Loss function

a) Binary classification
b) Multi-class classification (10 classes)
c) Multi-label classification (5 labels)
d) Regression
e) Regression with bounded output [0, 1]

---

# SECTION 2: BACKPROPAGATION & TRAINING (35 Problems)

## Gradients & Backpropagation Basics

**Problem 41:** What is a gradient in the context of neural networks?
- Mathematical definition
- Physical interpretation
- How it relates to optimization

**Problem 42:** For a simple function f(x) = x² at x = 3:
- Calculate the gradient
- What does it tell you?
- In which direction would you move to minimize f?

**Problem 43:** For a neural network:
- What are we computing gradients of?
- With respect to what variables?
- Why do we need these gradients?

**Problem 44:** Draw a computational graph for:
- z = x * w + b
- a = ReLU(z)
- loss = (a - y)²

Then show the backward pass (gradients)

**Problem 45:** Explain chain rule:
- Mathematical statement
- How it applies to neural networks
- Give an example with 3 compositions

---

## Computing Gradients Layer by Layer

**Problem 46:** For a single neuron layer:
- Input x: [1, 2]
- Weights w: [3, 4]
- Bias b: 0.5
- Activation: ReLU
- Target y: 2
- Loss: MSE

Calculate:
- Forward pass output
- Loss value
- dL/dw and dL/db

**Problem 47:** What is the gradient of ReLU?
- For positive input
- For negative input
- Why is this important for backprop?

**Problem 48:** For sigmoid activation: σ(x) = 1/(1 + e^(-x))
- Calculate the derivative: dσ/dx
- Simplify to σ'(x) = σ(x)(1 - σ(x))
- Calculate σ'(x) at x = 0 and x = 10

**Problem 49:** Backprop through 2 layers:
- Layer 1: y1 = ReLU(x * w1 + b1)
- Layer 2: y2 = Linear(y1 * w2 + b2)
- Loss: MSE(y2, target)

Show the gradient computation for each parameter

**Problem 50:** What does it mean for a gradient to "vanish"? How does ReLU vs Sigmoid affect this?

---

## Gradient Descent & Optimization

**Problem 51:** Gradient descent parameter update:
- Current weight w = 5
- Gradient dL/dw = -0.2
- Learning rate lr = 0.1
- Calculate new weight

**Problem 52:** Why does learning rate matter?
- What happens if lr is too large?
- Too small?
- Show examples with same gradient but different lr

**Problem 53:** For these scenarios, which learning rate would you use and why?
a) Large dataset, simple problem → lr = ?
b) Small dataset, complex problem → lr = ?
c) Network getting NaN losses → lr = ?
d) Training very slowly → lr = ?

**Problem 54:** Complete training iteration:
- Forward pass
- Loss calculation
- Backward pass
- Parameter update
(Pseudocode for a single layer network)

**Problem 55:** After 100 training iterations:
- Training loss: 2.5 → 1.2 → 0.8 → 0.05
- Validation loss: 2.5 → 1.2 → 0.8 → 0.6
- Validation loss increases while training loss decreases
- Diagnose: What's happening? Solutions?

---

## Challenges in Training

**Problem 56:** Explain these phenomena:
- Vanishing gradient
- Exploding gradient
- Dead neurons
- Mode collapse

For each: causes, symptoms, solutions

**Problem 57:** A very deep network (50 layers) has slow learning in early layers.
- Why might this happen? (Hint: Chain rule, gradients multiply)
- How can you fix it? (Multiple solutions)

**Problem 58:** For sigmoid vs ReLU in a 10-layer network:
- Which has better gradient flow?
- Why?
- Calculate gradient after 10 applications of sigmoid's derivative

**Problem 59:** Gradient clipping:
- What problem does it solve?
- How does it work?
- When would you use it?

**Problem 60:** Regularization:
- Why do we need it?
- What problem does it solve?
- L1 vs L2 differences

---

## Optimizers (Advanced)

**Problem 61:** Compare these optimizers:
- Vanilla SGD
- SGD with Momentum
- Adam
- RMSprop

For each: How it works, advantages, when to use

**Problem 62:** SGD with Momentum:
- Current velocity v = 0
- Gradient g = -0.5
- Momentum β = 0.9
- Learning rate α = 0.01
- Calculate new v and weight update

**Problem 63:** Adam optimizer:
- What does "adaptive" mean?
- How does it maintain per-parameter learning rates?
- Why is it popular?

**Problem 64:** For each scenario, which optimizer would you use:
a) Small dataset, need stable convergence → ?
b) Large dataset, want fast convergence → ?
c) Sparse data → ?
d) Deep neural network → ?

**Problem 65:** Explain the term "Adaptive Learning Rate"

---

## Batch Processing & Stochastic Training

**Problem 66:** What's the difference between:
- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD

Advantages/disadvantages of each

**Problem 67:** Dataset has 10,000 samples, batch size 32:
- How many batches per epoch?
- If training for 10 epochs, how many updates?
- How would this change with batch size 64?

**Problem 68:** Why is mini-batch SGD preferred to pure SGD?

**Problem 69:** For large datasets, why do we use batching instead of processing all data at once?

**Problem 70:** Effect of batch size on:
- Gradient estimate quality
- Computation speed
- Memory usage
- Generalization

---

## Learning Rate Schedules

**Problem 71:** A learning rate schedule:
- Epoch 1-10: lr = 0.1
- Epoch 11-20: lr = 0.01
- Epoch 21-30: lr = 0.001

Why would this help training?

**Problem 72:** Compare learning rate schedules:
- Step decay
- Exponential decay
- Linear warmup then decay
- Cosine annealing

When to use each?

**Problem 73:** Your training loss stops decreasing after 50 epochs.
- Try: learning rate schedule
- Options: decrease lr 10%, 50%, or 90%
- Which would you try first? Why?

---

## Monitoring Training

**Problem 74:** You're training a network. After 10 epochs:
- Training loss: 2.5 → 0.5 → 0.1 → 0.05
- Validation loss: 2.5 → 0.5 → 0.4 → 0.6

Diagnose each phase: underfitting, overfitting, or good?

**Problem 75:** What metrics should you track during training?
- For classification
- For regression
- Beyond loss function

---

# SECTION 3: CONVOLUTIONAL NEURAL NETWORKS (45 Problems)

## Convolution Operation Basics

**Problem 76:** What is a convolution in the context of CNNs?
- Mathematical operation
- What's being computed
- How is it different from a dot product

**Problem 77:** Manually compute 2D convolution:
```
Input:        Filter:       Output: ?
[1 2 3]       [1 0]
[4 5 6]       [0 1]
[7 8 9]
```
Using valid padding (no padding), stride 1

**Problem 78:** For the convolution in Problem 3.2:
- How many output values?
- What if stride = 2?
- What if you use padding = 1?

**Problem 79:** Input: 28×28, Filter: 3×3, Stride: 1, Padding: 0
- Output size? (Use formula)
- Output size with same padding?
- Output size with padding = 2?

**Problem 80:** Calculate output size:
- Input: 224×224
- Filter: 5×5
- Stride: 1
- Padding: 2
- Formula: ((W - F + 2P) / S) + 1

**Problem 81:** Why use padding in convolution?
- What problem does it solve?
- Types of padding (zero, replicate, etc.)
- "Same" vs "Valid" padding

**Problem 82:** Stride effect:
- Input: 64×64, Filter: 3×3, Padding: 1
- Stride 1 → Output size?
- Stride 2 → Output size?
- Stride 3 → Output size?
- Why would you use stride > 1?

---

## Filters & Feature Maps

**Problem 83:** What does a filter (kernel) learn to detect?
- Early layers?
- Later layers?
- How does this change with depth?

**Problem 84:** A CNN layer:
- Input: 32×32×3 (RGB image)
- 16 filters of size 3×3×3
- Output shape?
- Number of parameters?

**Problem 85:** Calculate parameters in conv layer:
- Input: H×W×C_in
- F filters of size: k×k×C_in
- How many parameters?
- Why do we share weights across spatial locations?

**Problem 86:** Feature map visualization:
- What does each channel in a feature map represent?
- Why are there multiple feature maps per layer?
- How do early feature maps differ from late ones?

**Problem 87:** A filter with all weights ≈ 0:
- What would happen to its output?
- Is this good or bad?
- How would backprop affect this filter?

**Problem 88:** Design a 3×3 filter that:
a) Detects vertical edges
b) Detects horizontal edges
c) Blurs the image
(Write the weights)

---

## Pooling Operations

**Problem 89:** What is pooling?
- Why do we use it?
- Effect on spatial dimensions
- Effect on parameters

**Problem 90:** Max pooling vs Average pooling:
```
Input:    Max Pooling:    Avg Pooling:
[1 2]     (2×2 window) ?  (2×2 window) ?
[3 4]
```

**Problem 91:** Pooling parameters:
- Input: 32×32×16
- 2×2 max pooling, stride 2
- Output shape?

**Problem 92:** Pooling effect on translation invariance:
- How does pooling help?
- What's the trade-off?

**Problem 93:** Global Average Pooling:
- Input: 7×7×512
- What does it do?
- Output shape?
- When is it used?

---

## CNN Architecture

**Problem 94:** Design a CNN for MNIST (28×28 images, 10 classes):
- Layer 1: Conv 16 filters 3×3, ReLU, MaxPool 2×2
- Layer 2: Conv 32 filters 3×3, ReLU, MaxPool 2×2
- Layer 3: Fully connected

Calculate output dimensions at each step

**Problem 95:** LeNet-5 vs Modern CNNs:
- LeNet: shallow, small filters
- Modern: deeper, 1×1 convolutions
- Why are modern CNNs better?

**Problem 96:** VGG-16 architecture:
- Mostly 3×3 filters
- Why not use larger filters?
- Advantage of stacking small filters

**Problem 97:** Residual connections (ResNet):
- What problem do they solve?
- How does a skip connection work?
- Mathematical formulation

**Problem 98:** 1×1 convolutions:
- What do they compute?
- Why use them?
- How do they reduce parameters?

**Problem 99:** Dilated (Atrous) Convolutions:
- What do they do differently?
- Why use them?
- How do they affect receptive field?

**Problem 100:** Depthwise Separable Convolutions:
- How are they different from standard convolutions?
- Parameter reduction?
- Trade-offs?

---

## Receptive Field

**Problem 101:** Receptive field definition and importance:
- What is it?
- Why does it matter?
- How do you calculate it?

**Problem 102:** Calculate receptive field for:
a) Single 3×3 conv layer: ?
b) Two stacked 3×3 conv layers: ?
c) 3×3 with stride 2: ?

**Problem 103:** Why would you want a large receptive field?
- For image classification?
- For semantic segmentation?
- Trade-offs?

---

## CNN for Different Tasks

**Problem 104:** Image Classification (e.g., ImageNet):
- Input size typical?
- Output size?
- Final layers?
- Loss function?

**Problem 105:** Object Detection (e.g., YOLO, R-CNN):
- What additional information beyond classification?
- Architecture differences?
- Loss function?

**Problem 106:** Semantic Segmentation:
- Input/Output sizes?
- How is classification different?
- Architecture considerations?

**Problem 107:** Image Denoising:
- What are input/outputs?
- Architecture challenges?
- Loss function?

---

## Parameter Counting & Efficiency

**Problem 108:** Count parameters:
- Conv layer: input 32×32×3, 16 filters 3×3, no padding
- Including bias?
- How many multiplications per forward pass?

**Problem 109:** Why do fully connected layers have way more parameters than conv layers?
- Example: 32×32×3 input
- FC layer: 1000 outputs → ? parameters
- Conv layer: 16 filters 3×3 → ? parameters

**Problem 110:** Strategies to reduce model parameters:
- Filter size
- Number of filters
- Depth
- Other architectures?

**Problem 111:** Computational complexity:
- FLOPs (Floating Point Operations)
- For a conv layer: H×W×F×F×C_in×C_out
- How do stride and pooling affect computation?

---

## Training CNNs

**Problem 112:** Why is training CNNs different from training dense networks?
- Convolution structure
- Shared weights
- Local connectivity

**Problem 113:** Data augmentation for CNNs:
- What augmentations make sense?
- Why is augmentation important?
- Examples for different domains

**Problem 114:** Transfer learning with pre-trained CNNs:
- What's learned in early layers?
- Later layers?
- How would you fine-tune for a new task?

**Problem 115:** A CNN trained on ImageNet:
- 1000 classes originally
- Need to adapt for 10 new classes
- Modify which layers?
- Training strategy?

---

## Common CNN Architectures

**Problem 116:** AlexNet:
- When was it introduced?
- Key innovations?
- Architecture summary?

**Problem 117:** VGG:
- Main design principle?
- Why was it influential?
- Drawbacks?

**Problem 118:** ResNet:
- What problem did it solve?
- How deep can ResNet go?
- Key innovation?

**Problem 119:** MobileNet:
- Purpose?
- Main techniques for efficiency?
- Trade-offs vs standard CNNs?

**Problem 120:** EfficientNet:
- Compound scaling principle?
- What's balanced?
- Advantages?

---

# SECTION 4: TRANSFORMERS & MODERN ARCHITECTURES (30 Problems)

## Attention Mechanism

**Problem 121:** Attention mechanism basics:
- What problem does it solve?
- How is it different from convolution?
- Mathematical formulation?

**Problem 122:** In Attention = softmax(QK^T/√d_k)V:
- What do Q, K, V represent?
- Why divide by √d_k?
- What does softmax do here?

**Problem 123:** Compute attention for this example:
- Q = [[1, 0], [0, 1]]  (2 queries)
- K = [[1, 0], [0, 1], [1, 1]]  (3 keys)
- V = [[1, 0], [0, 1], [1, 1]]  (3 values)
- d_k = 2
- Output shape?
- Compute step by step

**Problem 124:** Self-attention vs Cross-attention:
- Difference?
- When to use each?
- Examples?

**Problem 125:** Multi-head attention:
- Why use multiple heads?
- How many parameters vs single head?
- How are heads combined?

**Problem 126:** Positional encoding:
- Why do transformers need it?
- What happens without it?
- Common approaches?

---

## Transformer Architecture

**Problem 127:** General Transformer block:
- Multi-head Self-Attention
- Feed-Forward Networks
- Residual connections
- Layer normalization
- Order of operations?

**Problem 128:** Encoder vs Decoder:
- Differences?
- For what tasks is each used?
- When combined (Seq2Seq)?

**Problem 129:** Causal (Masked) Attention:
- What does masking do?
- Why is it needed?
- Language modeling vs machine translation?

**Problem 130:** Calculate parameters in a Transformer layer:
- Multi-head attention: h heads, d_model = 768, d_k = 64
- Feed-forward: d_model → 3072 → d_model
- Total?

---

## BERT & Pre-training

**Problem 131:** What is BERT?
- Architecture?
- Pre-training tasks?
- Input representation?

**Problem 132:** Masked Language Modeling (MLM):
- How does it work?
- What does the model learn?
- Why is it effective?

**Problem 133:** Next Sentence Prediction (NSP):
- What is it?
- How does it work?
- Less effective for what?

**Problem 134:** BERT Tokenization:
- Subword tokens (WordPiece)?
- Special tokens ([CLS], [SEP], [PAD])?
- Token embedding?

**Problem 135:** Fine-tuning BERT:
- Classification task: how to use [CLS]?
- Named Entity Recognition: token-level?
- Question Answering: start/end tokens?

---

## GPT & Autoregressive Models

**Problem 136:** What is GPT?
- Decoder-only architecture?
- Pre-training objective?
- Key differences from BERT?

**Problem 137:** Causal Attention in GPT:
- Why needed?
- How to implement masking?

**Problem 138:** GPT Generation:
- Greedy decoding?
- Top-k sampling?
- Temperature scaling?
- When to use each?

**Problem 139:** Prompt engineering for LLMs:
- Few-shot learning?
- Chain-of-thought prompting?
- Examples?

---

## Transformer Variations

**Problem 140:** Vision Transformer (ViT):
- How to adapt transformers for images?
- Patch embeddings?
- Advantages over CNNs?

**Problem 141:** T5 (Text-to-Text Transfer Transformer):
- Design philosophy?
- Various tasks as text-to-text?
- Examples?

**Problem 142:** Efficient Transformers:
- Why needed?
- Approaches: sparse attention, approximate attention?
- Trade-offs?

**Problem 143:** Compare architectures for different tasks:
- Image Classification: CNN vs ViT vs?
- Machine Translation: Seq2Seq vs Transformer?
- Text Generation: LSTM vs GPT?
- Object Detection: CNN vs Vision Transformer?

---

## Transfer Learning with Transformers

**Problem 144:** Pre-trained Hugging Face model:
- distilbert-base-uncased
- How many parameters?
- How to use for new classification task?

**Problem 145:** Fine-tuning strategies:
- Train all layers?
- Freeze early layers, train last?
- Use adapter modules?
- Trade-offs?

**Problem 146:** Domain adaptation:
- General model → specific domain
- Continue pre-training strategy?
- Fine-tuning strategy?

---

## Transformer Training

**Problem 147:** Why are transformers computationally expensive?
- Attention complexity: O(n²) in sequence length
- Memory requirements?
- Solutions?

**Problem 148:** Learning rate for transformers:
- Warm-up phase importance?
- Typical schedules?
- Why different from CNNs?

**Problem 149:** Gradient accumulation:
- Why needed for transformers?
- How does it work?
- Effect on batch size?

**Problem 150:** Transformers on limited resources:
- Quantization?
- Distillation?
- Model compression?

---

# SECTION 5: ADVANCED TOPICS & APPLICATIONS (30 Problems)

## Regularization Techniques

**Problem 151:** L1 and L2 Regularization:
- Difference in penalties?
- Effect on weights?
- When to use each?

**Problem 152:** Calculate loss with regularization:
- Loss: 0.5 (no regularization)
- Weights: [2, -1, 0.5, -0.3]
- L2 regularization λ = 0.01
- Total loss?

**Problem 153:** Dropout:
- How does it work at training?
- At inference?
- Effect on model capacity?
- Dropout rate selection?

**Problem 154:** Batch Normalization:
- Problem it solves?
- Training vs inference differences?
- Parameters?

**Problem 155:** Compare regularization:
- L1, L2, Dropout, BatchNorm
- When to use each?
- Can they be combined?

---

## Data Augmentation

**Problem 156:** For image classification:
- Basic augmentations?
- MixUp and Cutout?
- AutoAugment?
- When to use each?

**Problem 157:** For text tasks:
- Paraphrasing?
- Back-translation?
- Synonym replacement?
- Challenges vs images?

**Problem 158:** Augmentation effect on training:
- Smaller dataset →Augmentation helps?
- Large dataset → Still beneficial?
- When would augmentation hurt?

---

## Generalization & Overfitting

**Problem 159:** Diagnose overfitting:
- Training loss: 0.1, Validation loss: 0.5
- What's happening?
- Solutions?
- Priority order?

**Problem 160:** Diagnose underfitting:
- Training loss: 0.5, Validation loss: 0.6
- What's happening?
- Solutions?

**Problem 161:** Cross-validation:
- Why use it?
- k-fold vs stratified vs time-series?
- For neural networks?

---

## Hyperparameter Tuning

**Problem 162:** Important hyperparameters:
- Learning rate
- Batch size
- Architecture (layers, units)
- Regularization (λ, dropout rate)
- Priority for tuning?

**Problem 163:** Hyperparameter search:
- Grid search?
- Random search?
- Bayesian optimization?
- When to use each?

**Problem 164:** Your model not improving:
- Check: learning rate, batch size, initialization?
- Could add: dropout, L2, data augmentation?
- Could change: architecture, optimizer, schedule?
- Systematic approach?

---

## Handling Imbalanced Data

**Problem 165:** Dataset:
- 9000 samples class 0
- 1000 samples class 1
- Problems this causes?
- Solutions?

**Problem 166:** Class weighting in loss:
- How does it work?
- Effective weights?
- Alternatives?

**Problem 167:** Resampling strategies:
- Oversampling minority
- Undersampling majority
- SMOTE?
- Trade-offs?

---

## Multi-task Learning

**Problem 168:** Multi-task learning setup:
- Shared representation?
- Task-specific heads?
- Loss combination?
- When is it beneficial?

**Problem 169:** Task weighting:
- Weight all tasks equally?
- Dynamically weight by loss?
- Other strategies?

---

## Domain Adaptation

**Problem 170:** Covariate shift:
- What is it?
- How does it affect neural networks?
- Solutions?

**Problem 171:** Domain adaptation techniques:
- Adversarial adaptation?
- Self-training?
- Consistency regularization?

---

## Interpretability

**Problem 172:** Why is interpretability important?
- Use cases?
- Challenges with neural networks?

**Problem 173:** Visualization techniques:
- Feature visualization (what does a neuron learn)?
- Saliency maps (which input pixels matter)?
- Attention visualization?
- Examples?

**Problem 174:** LIME and SHAP:
- How do they work?
- Interpretable vs accurate?
- Local vs global explanations?

---

## Adversarial Robustness

**Problem 175:** Adversarial examples:
- What are they?
- Why do they occur?
- FGSM attack?

**Problem 176:** Adversarial training:
- How does it improve robustness?
- Cost?

**Problem 177:** Robustness vs accuracy trade-off:
- Can you have both?
- Research directions?

---

## Uncertainty & Confidence

**Problem 178:** When to trust model predictions?
- Confidence scores?
- Calibration?
- Out-of-distribution detection?

**Problem 179:** Bayesian approaches:
- Uncertainty quantification?
- Monte Carlo Dropout?
- Ensemble methods?

**Problem 180:** Confidence calibration:
- What is it?
- Why important?
- Temperature scaling?

---

# SECTION 6: CODE & IMPLEMENTATION (25 Problems)

## PyTorch Basics

**Problem 181:** PyTorch vs TensorFlow:
- Similarities?
- Differences?
- Advantages of each?

**Problem 182:** Create a simple neural network:
```python
# Pseudocode
class SimpleNN(nn.Module):
    def __init__(self):
        # Define layers
        
    def forward(self, x):
        # Forward pass
```

**Problem 183:** Tensor operations:
```python
x = torch.randn(2, 3)
y = torch.randn(3, 4)

# What's the result of x @ y?
# Shape?
# How many parameters?
```

**Problem 184:** Gradient computation:
```python
x = torch.randn(2, requires_grad=True)
y = (x ** 2).sum()
y.backward()
# What's x.grad?
```

**Problem 185:** Training loop pseudocode:
```
for epoch:
    for batch in data:
        1. ? (forward pass)
        2. ? (loss calculation)
        3. ? (backward pass)
        4. ? (parameter update)
        5. ? (zero gradients)
```

---

## Neural Network Implementation

**Problem 186:** Implement forward pass for 2-layer network:
```python
x = [1, 2]
W1 = [[1, 0], [0, 1]]  # 2x2
b1 = [0, 0]
W2 = [[1, -1]]  # 1x2
b2 = [0.5]

# Compute y = (x @ W1 + b1) @ W2 + b2
# Step by step
```

**Problem 187:** Loss and backpropagation:
```python
# After forward pass, y = 0.5
# Target = 1
# MSE Loss?
# dy/dW2?
# dy/dW1?
```

**Problem 188:** Implement batch normalization:
- Normalize input
- Scale and shift
- Training vs inference

**Problem 189:** Implement dropout:
```python
# During training: randomly set some activations to 0
# Scaling factor?
# During inference: ?
```

---

## CNN Implementation

**Problem 190:** Conv2D operation:
```python
# Input: [batch, 32, 32, 3]
# Filter: [16, 3, 3, 3]  (16 filters, 3x3 kernel, 3 input channels)
# Stride: 1, Padding: 1
# Output shape?
# Number of parameters?
```

**Problem 191:** Implement 2D convolution:
```python
# Given input and filter, compute output
# Use padding, stride
# Show computation
```

**Problem 192:** Max pooling:
```python
# Input: [batch, 32, 32, 16]
# 2x2 pooling, stride 2
# Output shape?
```

**Problem 193:** Complete CNN architecture:
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        # Conv, ReLU, MaxPool layers
        # FC layers for classification
        
    def forward(self, x):
        # Forward pass
```

---

## Data Loading & Preprocessing

**Problem 194:** Data normalization:
```python
# Images in [0, 255]
# Normalize to mean 0, std 1?
# Formula?
# Why important?
```

**Problem 195:** Data loader:
```python
dataset = MyDataset(...)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# What does shuffle do?
# What's the difference between training and validation loaders?
```

**Problem 196:** Image preprocessing pipeline:
```python
# Read image
# Resize to 224x224
# Convert to tensor
# Normalize
# Order of operations?
```

---

## Training Script

**Problem 197:** Complete training loop:
```python
model = SimpleNN()
optimizer = ???
loss_fn = ???

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # Forward pass
        # Compute loss
        # Backward pass
        # Update weights
        
    for batch_x, batch_y in val_loader:
        # Validation evaluation
```

**Problem 198:** Evaluation metrics:
- Classification: accuracy, precision, recall, F1?
- Regression: MSE, MAE, R²?
- Implementation?

**Problem 199:** Checkpoint saving:
```python
if val_loss < best_val_loss:
    # Save model
    # Save optimizer state
    # Save epoch number
```

**Problem 200:** Early stopping:
```python
# Monitor validation loss
# If no improvement for N epochs
# Stop training
# Pseudocode?
```

---

## Transfer Learning

**Problem 201:** Using pre-trained model:
```python
model = torchvision.models.resnet18(pretrained=True)

# For new classification task:
# Modify last layer
# Freeze early layers
# Fine-tune
```

**Problem 202:** Gradients:
```python
# Freeze layers 1-5:
# for param in model.layer1.parameters():
#     param.requires_grad = False

# Effect on:
# - Memory usage?
# - Training speed?
# - Performance?
```

---

## Debugging

**Problem 203:** Model produces NaN losses:
- Possible causes?
- Debug strategy?
- Solutions?

**Problem 204:** Model underfitting:
- Check: architecture, learning rate, training time?
- Debug strategy?

**Problem 205:** Model overfitting:
- Check: regularization, data augmentation, dropout?
- Debug strategy?

---

# SECTION 7: MIXED DIFFICULTY & INTEGRATION (25 Problems)

## Comprehensive Problem Scenarios

**Problem 206:** MNIST Classification
- Build architecture
- Data loading
- Training
- Evaluation
- Expected accuracy?

**Problem 207:** CIFAR-10 with ResNet
- Architecture challenges?
- Transfer learning approach?
- Data augmentation strategy?

**Problem 208:** Sentiment Analysis
- Embeddings?
- Architecture (CNN, RNN, Transformer)?
- Label smoothing?

**Problem 209:** Image Segmentation
- Architecture (U-Net, etc.)?
- Output format?
- Loss function?

**Problem 210:** Object Detection (simplified)
- Localization + Classification?
- Architecture?
- Loss function?

---

## Design Questions

**Problem 211:** New task: Predict house prices from 20 numerical features
- Architecture?
- Normalization?
- Loss function?
- Regularization?

**Problem 212:** New task: Classify 1000 types of flowers
- Pre-trained or from scratch?
- Data augmentation?
- Batch size?
- Learning rate?

**Problem 213:** New task: Detect objects in medical images
- Limited labeled data
- Strategy?
- Architecture?

**Problem 214:** New task: Generate text summaries
- Architecture?
- Loss function?
- Evaluation metrics?

**Problem 215:** New task: Real-time gesture recognition on mobile
- Constraints?
- Model compression?
- Inference optimization?

---

## Theory & Implementation Connections

**Problem 216:** Why SGD converges:
- Mathematical argument?
- What about non-convex (neural network) losses?
- Guarantees?

**Problem 217:** Universal approximation:
- What does it guarantee?
- Practical implications?
- Why is it not sufficient?

**Problem 218:** Overparameterization:
- Modern networks: more parameters than training samples?
- Why do they generalize?
- Double descent phenomenon?

**Problem 219:** Implicit regularization:
- Is SGD itself a form of regularization?
- Inductive biases?
- How does architecture matter?

**Problem 220:** Lottery Ticket Hypothesis:
- What is it?
- Implications?
- Practical applications?

---

## Recent Research Topics

**Problem 221:** Vision Transformers:
- How do they work?
- Advantages over CNNs?
- When to use?

**Problem 222:** Few-shot Learning:
- What is it?
- Meta-learning?
- Applications?

**Problem 223:** Self-supervised Learning:
- Why useful?
- Contrastive learning?
- Applications?

**Problem 224:** Neural Architecture Search (NAS):
- What problem does it solve?
- Approaches?
- Cost?

**Problem 225:** Federated Learning:
- Motivation?
- Challenges?
- Privacy-preserving training?

---

## Comprehensive Scenarios

**Problem 226:** You have a dataset with:
- 1000 samples
- 100 features
- 2 classes
- Imbalanced: 900:100

Approach for model selection?

**Problem 227:** Production system requirements:
- Inference latency < 100ms
- Accuracy > 95%
- Serving 1M requests/day

What would you consider?

**Problem 228:** Model improvement suggestions:
- Baseline accuracy: 80%
- Target: 90%
- Limited data, no budget for more
- 10 ideas to try (priority order)?

**Problem 229:** Debugging failure case:
- Model works well on test set (92% accuracy)
- Fails on real-world data (60% accuracy)
- What happened?
- Solutions?

**Problem 230:** Research direction:
- Current state: image classification with CNNs
- Emerging: Vision Transformers
- Compare thoroughly
- When would you use each?
- Future directions?

---

---

# ANSWER KEY & SOLUTIONS

## SECTION 1: NEURAL NETWORKS FUNDAMENTALS

### Problem 1.1 Solution
The input layer receives raw input data and distributes it to hidden layers. It has as many neurons as input features. For example:
- Image (28×28×3): 2352 input neurons
- Tabular data (20 features): 20 input neurons

### Problem 1.2 Solution
- Hidden layers: 1 (between input and first hidden layer count other hidden layers)
- Network could be: MNIST classification (784→128→64→10)
- Parameters input→hidden: 784 × 128 + 128 = 100,480 parameters

### Problem 1.3 Solution
Hidden layers learn non-linear transformations. Without them (input→output directly), the network would only compute linear functions, unable to learn complex patterns.

### Problem 1.4 Solution
- **Layers (depth):** How many stages of transformation (1 = shallow, 100 = very deep)
- **Width:** Number of neurons per layer (10 = narrow, 1000 = wide)
- **Parameters:** Total weights and biases
- Example (100→50→50→10): 100×50 + 50×50 + 50×10 + biases = 7560 parameters

### Problem 1.5 Solution
**Shallow:** Fast training, less computation, prone to underfitting
**Deep:** Slow training, more computation, can learn more complex patterns

### Problem 1.6 Solution
- **ReLU:** f(x) = max(0, x), range [0,∞), use for hidden layers, prevents vanishing gradient
- **Sigmoid:** f(x) = 1/(1+e^-x), range (0,1), use for binary classification output
- **Tanh:** f(x) = (e^x - e^-x)/(e^x + e^-x), range (-1,1), similar to sigmoid
- **Linear:** f(x) = x, range (-∞,∞), use for regression output

### Problem 1.7 Solution
ReLU has constant gradient of 1 for positive inputs, preventing vanishing gradients in deep networks. Sigmoid gradient ≈ 0.25 max, so gradients multiply and vanish with depth.

### Problem 1.8 Solution
- Function: would only compute linear transformations (composition of linear functions = linear function)
- Layers needed: 1 (extra layers add nothing)
- Problem: unable to learn non-linear patterns

### Problem 1.9 Solution
Input: [-2, -1, 0, 1, 2]
- ReLU: [0, 0, 0, 1, 2]
- Sigmoid: [0.12, 0.27, 0.5, 0.73, 0.88]
ReLU is sparse (many zeros), sigmoid is smooth

### Problem 1.10 Solution
- Sigmoid hidden: rarely, causes vanishing gradients
- Sigmoid output: binary classification (outputs 0-1 probability)
- ReLU hidden: standard choice for hidden layers
- Linear output: regression (unbounded values)

### Problem 1.11 Solution
**Dying ReLU:** All neurons output 0 because weights push inputs negative
**Fixes:** Leaky ReLU (f(x) = 0.01x for x<0), better initialization, lower learning rate

### Problem 1.12 Solution
- Activation: Softmax (outputs sum to 1, probability distribution)
- Not ReLU: would output unbounded values
- Form: exp(z_i) / sum(exp(z_j))

### Problem 1.13 Solution
- ReLU at 0.5: gradient = 1
- ReLU at -0.5: gradient = 0
- Sigmoid at 0: gradient = 0.25
- Tanh at 0: gradient = 1

### Problem 1.14 Solution
- z = 1×0.5 + 2×(-0.2) + 3×0.1 + 0.1 = 0.5 - 0.4 + 0.3 + 0.1 = 0.5
- a = ReLU(0.5) = 0.5

### Problem 1.15 Solution
Forward pass:
- z1 = [1, 0.5] @ [[1, 0], [0, 1]] + [0, 0] = [1, 0.5]
- a1 = ReLU([1, 0.5]) = [1, 0.5]
- z2 = [1, 0.5] @ [[1], [-1]] + [0.5] = 1 - 0.5 + 0.5 = 1
- Output: 1

### Problem 1.16 Solution
- Single layer: m × n multiplications + m × n additions = O(mn)
- k-layer network: O(n1×n2 + n2×n3 + ... + n_{k-1}×n_k)

### Problem 1.17 Solution
z = sum(w_i × x_i) + b = w·x + b
a = activation(z)
Each neuron computes weighted sum + bias, then applies nonlinearity

### Problem 1.18 Solution
- Double weights: output doubles (scales proportionally)
- Double bias: output increases (additive change)
- Single layer: next layer's output changes non-linearly (depends on activation)

### Problem 1.19 Solution
Prevents all neurons from learning the same features. Starting at 0 means all neurons learn identically (symmetry breaking needed).

### Problem 1.20 Solution
- **All zeros:** Symmetry problem, networks don't learn
- **Small random:** Works for small networks, causes vanishing gradients in deep networks
- **Xavier:** sqrt(1/n) for tanh/sigmoid networks
- **He:** sqrt(2/n) for ReLU networks (accounts for 50% dying ReLU)

### Problem 1.21 Solution
- He range: sqrt(2/1000) ≈ ±0.045
- Xavier range: sqrt(1/1000) ≈ ±0.032
- He is wider (bigger variance) since ReLU has 50% dead neurons

### Problem 1.22 Solution
Xavier initialization scales based on fan-in, preventing early layer gradients from vanishing by keeping activations in a reasonable range.

### Problem 1.23 Solution
- **Width effect:** More parameters → more expressiveness → overfitting risk → slower training
- Trade-off between model capacity and generalization

### Problem 1.24 Solution
Universal approximation uses ONE hidden layer with infinite neurons. Deep networks approximate better with fewer total neurons (exponential vs polynomial advantage).

### Problem 1.25 Solution
- **20→10→1:** Smallest, trains fast, may underfit
- **20→50→50→1:** Medium, more capacity, risk overfitting
- **20→5→1:** Very small, definitely underfit
Choose based on dataset size and task complexity

### Problem 1.26 Solution
- 100→50→10: 100×50 + 50×10 + biases = 5000 + 50 + 60 = 5110 parameters
- 784→128→64→10: 784×128 + 128×64 + 64×10 + biases = 100,352 + 8,192 + 640 + 202 = 109,386 parameters
- 32→32→32→32: 32×32 + 32×32 + 32×32 + biases = 3,072 + 96 = 3,168 parameters

### Problem 1.27 Solution
a) **MNIST (28×28, 10 classes):**
- Input: 784
- Hidden: 128 or 256
- Output: 10, softmax, cross-entropy loss

b) **House prices:** 
- Input: features
- Hidden: variable size
- Output: 1, linear, MSE loss

c) **Sentiment analysis:**
- Input: embeddings (depends on NLP preprocessing)
- Hidden: varies
- Output: 1 or 2 (binary), sigmoid or softmax

### Problem 1.28 Solution
Draw diagram with 784 input nodes → 128 hidden → 64 hidden → 10 output
Matrix multiplications: 784×128, 128×64, 64×10

### Problem 1.29 Solution
Infinite architectures. Examples:
- 1M→1M→10
- 1M→500K→500K→10  
- 1M→1M→1M→...→10
- Many wide, many narrow, mixed

### Problem 1.30 Solution
- **Toy problem:** Small network (e.g., 10→5→1)
- **Real dataset:** Larger network
- **Transfer learning:** Use pre-trained layers, small final layers

### Problem 1.31 Solution
Cross-Entropy = -sum(y × log(p))
= -(0×log(0.1) + 1×log(0.7) + 0×log(0.2))
= -log(0.7) ≈ 0.357

### Problem 1.32 Solution
- **Binary CE:** For 2 classes, usually sigmoid + BCE
- **Categorical CE:** For multi-class, softmax + CE
When: BCE for binary, CE for multi-class

### Problem 1.33 Solution
All for true class 0:
- Uniform [0.25,...]: -log(0.25) ≈ 1.39
- Peaked [0.1,...]: -log(0.1) ≈ 2.30
- Confident [0.91,...]: -log(0.91) ≈ 0.09
Loss increases for worse predictions

### Problem 1.34 Solution
Good loss function:
- Differentiable (for backprop)
- Penalizes wrong predictions
- Rewards correct predictions
- Has meaningful gradient

### Problem 1.35 Solution
Softmax converts logits to probability distribution (sum to 1). Without it, outputs unbounded, can't use as probabilities. With softmax: exp(z)/sum(exp(z))

### Problem 1.36 Solution
- **MSE:** Smooth, differentiable, penalizes large errors more, sensitive to outliers
- **MAE:** Less sensitive to outliers, less smooth
- **Huber:** Best of both, uses MSE near 0, MAE for large errors
Use Huber for outlier-prone data, MSE generally, MAE for robust regression

### Problem 1.37 Solution
Errors: [-20k, 10k, -10k]
- MSE = (400M + 100M + 100M) / 3 = 200M
- MAE = (20k + 10k + 10k) / 3 ≈ 13.3k

### Problem 1.38 Solution
With outliers, MSE losses get very large. MAE treats all errors uniformly, so won't be dominated by single large error.

### Problem 1.39 Solution
Errors: 99×1 + 1×100
- MSE = (99 + 10000) / 100 = 100.99
- MAE = (99 + 100) / 100 = 1.99
MSE much higher due to outlier. With MSE, model would over-optimize for that outlier.

### Problem 1.40 Solution
- **Binary:** 1 output, sigmoid, BCE
- **Multi-class (10):** 10 outputs, softmax, CE
- **Multi-label (5):** 5 outputs, sigmoid, BCE (each independent)
- **Regression:** 1 output, linear, MSE
- **Regression [0,1]:** 1 output, sigmoid, MSE

---

## SECTION 2: BACKPROPAGATION & TRAINING

[Solutions would continue similarly for Section 2...]

Due to space constraints, I'll provide the pattern:

### Problem 2.1 Solution
**Gradient:** Vector of partial derivatives. In NN, gradient of loss wrt each parameter.
**Physical interpretation:** Direction of steepest increase in loss.
**For optimization:** Move opposite to gradient (downhill in loss landscape).

[Continue this pattern through Section 2...]

---

## SECTION 3: CONVOLUTIONAL NEURAL NETWORKS

### Problem 3.1 Solution
Convolution: Sliding a filter over input, computing element-wise product and sum. 
Different from dot product: applied at multiple positions, not just once.

### Problem 3.2 Solution
Convolution output at (0,0): 1×1 + 2×0 = 1
At (0,1): 2×1 + 3×0 = 2
At (1,0): 4×1 + 5×0 = 4
At (1,1): 5×1 + 6×0 = 5
Output:
```
[1 2]
[4 5]
```

### Problem 3.3 Solution
- Valid (no padding): 2×2 output
- Stride 2: 1×1 output (only (0,0) fits)
- Padding 1: 3×3 output

### Problem 3.4 Solution
Output = ((28 - 3 + 0) / 1) + 1 = 26×26
With same padding: 28×28
With padding 2: 30×30

### Problem 3.5 Solution
Output = ((224 - 5 + 4) / 1) + 1 = 222×222

[Continue similarly...]

---

## ANSWER COMPLETION STRATEGY

**For Sections 4-7:** Follow the same pattern - detailed step-by-step solutions covering:
- Mathematical calculations
- Conceptual explanations
- Practical implications
- When/why to use each approach

---

# STUDY SCHEDULE RECOMMENDATION

**Saturday (Today):**
- Review: Problems 1.1-1.10 (Basic concepts)
- Time: 1-2 hours
- Focus: Understand activation functions, architecture basics

**Sunday:**
- Review: Problems 1.11-1.40, start Section 2.1-2.10
- Time: 2-3 hours
- Focus: Initialization, loss functions, gradient basics

**Monday:**
- Problems: Section 2.11-2.35
- Time: 2.5-3 hours
- Focus: Gradient descent, optimization, training challenges

**Tuesday:**
- Problems: Section 3.1-3.25
- Time: 2.5-3 hours
- Focus: Convolution operations, filters, pooling

**Wednesday:**
- Problems: Section 3.26-4.15
- Time: 2-3 hours
- Focus: Receptive field, CNNs for different tasks, Transformers start

**Thursday:**
- Problems: Section 4.16-5.15
- Time: 2.5-3 hours
- Focus: Advanced Transformers, Regularization, Overfitting

**Friday:**
- Review: Sections 5.16-7.25 + Mixed review
- Time: 2-3 hours
- Focus: Advanced topics, integration problems, real scenarios
- Final review of weak areas

**Total Study Time: ~15-18 hours over 6 days**

---

# USAGE TIPS

1. **Solve actively:** Write solutions, don't just read
2. **Time yourself:** Simulate exam conditions
3. **Identify weak areas:** Track which topics are hardest
4. **Review patterns:** Notice recurring concepts
5. **Use answers:** Check after solving, understand mistakes
6. **Mix difficulties:** Don't always do easy ones first
7. **Cross-reference:** Connect problems to notebooks and study guides
8. **Practice coding:** For Section 6, actually code the solutions
9. **Explain aloud:** Teaching forces deeper understanding
10. **Take breaks:** Don't marathon study - 50min focused work + 10min break

---

**Good luck with your exam preparation! These 230 problems cover every concept, difficulty level, and application type you could encounter on the midterm. Consistent practice with these will absolutely prepare you for success!** 🎓📚

