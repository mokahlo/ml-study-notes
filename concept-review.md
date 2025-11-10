# Concept Review - CEE 501 Midterm 2

## Table of Contents
1. [Neural Network Fundamentals](#neural-network-fundamentals)
2. [Convolutional Neural Networks](#convolutional-neural-networks)
3. [Support Vector Machines](#support-vector-machines)
4. [Decision Trees](#decision-trees)
5. [Training and Evaluation](#training-and-evaluation)

---

## Neural Network Fundamentals

### Basic Building Blocks

#### 1. Linear Transformation
```
z = w * x + b
```
- **w**: Weight (how much influence input has)
- **x**: Input value
- **b**: Bias (shifts the output)
- **z**: Linear combination (pre-activation)

**Intuition:** The weight scales the input, bias shifts it. This is like `y = mx + b` from algebra.

#### 2. Activation Functions

##### Sigmoid
```
σ(z) = 1 / (1 + e^(-z))
```
- **Range:** (0, 1)
- **Use:** Binary classification, output probabilities
- **Properties:**
  - Smooth, differentiable
  - Output interpretable as probability
  - Suffers from vanishing gradient

##### ReLU (Rectified Linear Unit)
```
ReLU(z) = max(0, z)
```
- **Range:** [0, ∞)
- **Use:** Hidden layers in deep networks
- **Properties:**
  - Computationally efficient
  - Helps with vanishing gradient
  - Can cause "dead neurons" (always output 0)

##### Tanh
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- **Range:** (-1, 1)
- **Use:** Hidden layers, RNNs
- **Properties:**
  - Zero-centered (better than sigmoid)
  - Still suffers from vanishing gradient

#### 3. Loss Functions

##### Squared Loss (Mean Squared Error)
```
L = 1/2 * (ŷ - y)²
```
- **Use:** Regression problems
- **Measures:** Average squared difference

##### Cross-Entropy Loss
```
L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```
- **Use:** Classification problems
- **Measures:** Difference between predicted and actual probability distributions

### Forward Propagation
**Process:** Input → Linear → Activation → Linear → ... → Output

**Example:**
```
Input layer: x = [x₁, x₂]
Hidden layer: h = σ(W₁·x + b₁)
Output layer: ŷ = σ(W₂·h + b₂)
```

### Backpropagation
**Purpose:** Calculate gradients to update weights

**Chain Rule Application:**
```
∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w
```

---

## Convolutional Neural Networks

### Key Concepts

#### 1. Convolution Operation
**Purpose:** Extract local features from input

**How it works:**
1. Slide a filter (kernel) over the input
2. Element-wise multiply filter with input patch
3. Sum all products and add bias
4. Store result in output feature map

**Visual Example:**
```
Input (4×4):        Filter (2×2):
[1  2  3  4]        [1  0]
[5  6  7  8]        [0  1]
[9  10 11 12]
[13 14 15 16]

Convolution at position (0,0):
[1  2]  ⊙  [1  0]  =  1×1 + 2×0 + 5×0 + 6×1 = 7
[5  6]     [0  1]
```

#### 2. Stride
**Definition:** Number of pixels to move filter

- **Stride = 1:** Overlapping windows, larger output
- **Stride = 2:** Skip pixels, smaller output

**Effect on output size:**
```
Larger stride → Smaller output
Stride = 1: More features extracted
Stride > 1: Downsampling
```

#### 3. Padding
**Purpose:** Control output size and preserve edge information

- **Valid (No Padding):** Output smaller than input
- **Same Padding:** Output same size as input
- **Formula:** Add `p` pixels of zeros around border

**Common patterns:**
```
Padding = 0: Output shrinks
Padding = (Filter-1)/2: Output size preserved (stride=1)
```

#### 4. Pooling
**Purpose:** Downsample, reduce parameters, provide translation invariance

##### Max Pooling
```
Input (4×4):        Pooling (2×2, stride=2):
[1  3  2  4]        [6  8]
[5  6  7  8]   →   [14 16]
[9  10 11 12]
[13 14 15 16]

Takes maximum value in each window
```

##### Average Pooling
- Takes average instead of maximum
- Smoother downsampling

#### 5. Output Size Formula
```
Output = ⌊(Input - Filter + 2×Padding) / Stride⌋ + 1
```

**Key Points:**
- ⌊ ⌋ means floor (round down)
- Don't forget the +1!
- Each dimension calculated separately

#### 6. Parameter Counting
```
Parameters = (Filter_H × Filter_W × Input_Depth + 1) × Num_Filters
                                                   ↑
                                               Bias term
```

**Example:**
- 3×3 filter, 3 input channels (RGB), 64 filters
- Params = (3 × 3 × 3 + 1) × 64 = 28 × 64 = 1,792

### Typical CNN Architecture
```
Input Image (32×32×3)
    ↓
Conv Layer (filter: 3×3, filters: 32, stride: 1, padding: 1)
    → Output: 32×32×32
    ↓
ReLU Activation
    ↓
Max Pooling (2×2, stride: 2)
    → Output: 16×16×32
    ↓
Conv Layer (filter: 3×3, filters: 64, stride: 1, padding: 1)
    → Output: 16×16×64
    ↓
ReLU Activation
    ↓
Max Pooling (2×2, stride: 2)
    → Output: 8×8×64
    ↓
Flatten → 4096 units
    ↓
Fully Connected Layer → 128 units
    ↓
Output Layer → 10 classes
```

---

## Support Vector Machines

### Core Concepts

#### 1. Decision Boundary (Hyperplane)
**Goal:** Find the best line/plane that separates classes

**Mathematical Form:**
```
f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w·x + b
```

**Classification Rule:**
- If `f(x) > 0`: Positive class
- If `f(x) < 0`: Negative class
- If `f(x) = 0`: On the boundary

#### 2. Margin
**Definition:** Distance between decision boundary and nearest data points

**SVM Objective:** Maximize the margin
- Larger margin → Better generalization
- Points on margin boundary are "support vectors"

#### 3. Linear vs Non-Linear
**Linear SVM:**
- Classes separable by straight line/plane
- Fast, simple

**Non-Linear SVM (Kernel Trick):**
- Transform data to higher dimensions
- Find linear separator in new space
- Common kernels: RBF, Polynomial

**Visual:**
```
Original Space:          Transformed Space:
   (Non-separable)         (Separable)
       ×                        ×
    ×  o  ×                  ×     ×
  ×  o  o  ×              o   o   o
    ×  o  ×                  |
       ×              -------+-------
                            Line!
```

#### 4. Soft vs Hard Margin
**Hard Margin:**
- Requires perfect separation
- No misclassifications allowed
- Not robust to outliers

**Soft Margin:**
- Allows some misclassifications
- Parameter C controls trade-off
- More practical for real data

---

## Decision Trees

### Concepts

#### 1. Tree Structure
```
           [Root: Feature A]
                /      \
         A < 5          A >= 5
            /              \
    [Feature B]        [Class: Yes]
       /    \
   B < 3   B >= 3
     /        \
[Class: No] [Class: Yes]
```

#### 2. Splitting Criteria

##### Entropy (Measure of Impurity)
```
E = -Σ p_i × log₂(p_i)
```
- **p_i:** Proportion of class i
- **Range:** 0 (pure) to 1 (maximum impurity for binary)

**Examples:**
```
All same class: E = 0 (pure)
50-50 split: E = 1 (maximum impurity)
```

##### Information Gain
```
IG = E(parent) - Σ (|child_i|/|parent|) × E(child_i)
```
- Measures reduction in entropy
- Higher IG → Better split

#### 3. Gini Impurity (Alternative to Entropy)
```
Gini = 1 - Σ p_i²
```
- Similar to entropy
- Faster to compute
- Range: 0 (pure) to 0.5 (binary, maximum impurity)

#### 4. Stopping Criteria
- Maximum depth reached
- Minimum samples per leaf
- No further information gain
- All samples same class

#### 5. Overfitting
**Problem:** Tree memorizes training data

**Solutions:**
- **Pruning:** Remove branches that don't improve validation performance
- **Max depth:** Limit tree size
- **Min samples split:** Require minimum samples to split
- **Random Forest:** Use ensemble of trees

---

## Training and Evaluation

### Training Process

#### 1. Gradient Descent
```
w_new = w_old - α × ∂L/∂w
```
- **α (learning rate):** Step size
- **∂L/∂w:** Gradient (direction of steepest increase)

**Variants:**
- **Batch GD:** Use all data
- **Stochastic GD (SGD):** Use one sample
- **Mini-batch GD:** Use small batch (most common)

#### 2. Learning Rate
- **Too small:** Slow convergence
- **Too large:** Oscillation, divergence
- **Just right:** Fast, stable convergence

#### 3. Epochs
**Definition:** One complete pass through training data
- More epochs → Better fit (but risk overfitting)
- Monitor validation loss to detect overfitting

### Evaluation Metrics

#### 1. Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Simple, intuitive
- Misleading for imbalanced datasets

#### 2. Precision
```
Precision = TP / (TP + FP)
```
- Of all positive predictions, how many correct?
- Important when false positives costly

#### 3. Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- Of all actual positives, how many found?
- Important when false negatives costly

#### 4. F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balanced measure

#### 5. Confusion Matrix
```
                Predicted
              Neg    Pos
Actual  Neg   TN     FP
        Pos   FN     TP
```

#### 6. ROC Curve & AUC
- **ROC:** True Positive Rate vs False Positive Rate
- **AUC:** Area under ROC curve
- **Range:** 0.5 (random) to 1.0 (perfect)

### Overfitting vs Underfitting

#### Overfitting
- **Signs:** High training accuracy, low validation accuracy
- **Cause:** Model too complex
- **Solutions:**
  - More training data
  - Regularization (L1, L2)
  - Dropout
  - Early stopping
  - Reduce model complexity

#### Underfitting
- **Signs:** Low training and validation accuracy
- **Cause:** Model too simple
- **Solutions:**
  - Increase model complexity
  - Add features
  - Train longer
  - Reduce regularization

---

## Quick Reference Tables

### Activation Functions Comparison
| Function | Range | Advantages | Disadvantages |
|----------|-------|------------|---------------|
| Sigmoid | (0,1) | Smooth, probability | Vanishing gradient |
| Tanh | (-1,1) | Zero-centered | Vanishing gradient |
| ReLU | [0,∞) | Fast, no vanishing | Dead neurons |
| Leaky ReLU | (-∞,∞) | No dead neurons | Arbitrary negative slope |

### CNN Layer Effects
| Operation | Size Effect | Parameter Count | Purpose |
|-----------|-------------|-----------------|---------|
| Convolution | Depends on stride/padding | H×W×D×F + F | Feature extraction |
| Max Pooling | Reduces | 0 | Downsampling |
| Fully Connected | Fixed output | Input×Output + Output | Classification |

### When to Use What

#### Choose CNN when:
- Data has spatial structure (images)
- Translation invariance needed
- Local patterns important

#### Choose SVM when:
- Clear margin between classes
- High-dimensional space
- Small to medium datasets

#### Choose Decision Trees when:
- Interpretability crucial
- Mixed feature types
- Non-linear relationships
- No need for scaling/normalization

#### Choose Neural Networks when:
- Large datasets available
- Complex patterns
- End-to-end learning desired
- Computational resources available

---

## Common Pitfalls

### Mathematical Errors
- ❌ Forgetting +1 in output size formula
- ❌ Using ln instead of log₂ for entropy
- ❌ Not including bias in parameter count
- ❌ Confusing stride with filter size

### Conceptual Mistakes
- ❌ Thinking more layers always better
- ❌ Ignoring validation performance
- ❌ Using accuracy for imbalanced data
- ❌ Not normalizing input data

### Implementation Issues
- ❌ Wrong initialization (all zeros)
- ❌ Learning rate too high/low
- ❌ Forgetting to shuffle training data
- ❌ Not using separate validation set

---

## Study Strategies

1. **Understand Formulas:** Don't just memorize - know when/why to use each
2. **Work Examples:** Practice calculations by hand
3. **Visualize:** Draw networks, decision boundaries, trees
4. **Connect Concepts:** How do different topics relate?
5. **Test Yourself:** Explain concepts without notes

Good luck with your midterm! 🎓
