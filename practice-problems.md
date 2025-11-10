# Practice Problems - CEE 501 Midterm 2

## Problem Set 1: CNN Dimensions

### Problem 1.1: Basic Convolution
**Given:**
- Input: 8x8 image
- Filter: 3x3
- Stride: 1
- Padding: 0

**Questions:**
1. What is the output size?
2. If we use 16 filters, how many parameters are there?

**Solution:**
```
1. Output = [(8 - 3 + 2*0) / 1] + 1 = 6x6
2. Parameters = (3 × 3 × 1 + 1) × 16 = 10 × 16 = 160
```

---

### Problem 1.2: Convolution with Padding
**Given:**
- Input: 10x10 image
- Filter: 5x5
- Stride: 2
- Padding: 2

**Questions:**
1. What is the output size?
2. With 32 filters and input depth of 3 (RGB), how many parameters?

**Solution:**
```
1. Output = [(10 - 5 + 2*2) / 2] + 1 = [(10 - 5 + 4) / 2] + 1 = [9/2] + 1 = 4 + 1 = 5x5
2. Parameters = (5 × 5 × 3 + 1) × 32 = 76 × 32 = 2,432
```

---

### Problem 1.3: Max Pooling
**Given:**
- Input: 16x16 feature map
- Pooling window: 2x2
- Stride: 2
- No padding

**Question:** What is the output size?

**Solution:**
```
Output = [(16 - 2 + 0) / 2] + 1 = 8x8
```

---

### Problem 1.4: Multiple Convolution Layers
**Given:**
- Input: 32x32x3 (RGB image)
- Conv1: 5x5 filters, 16 filters, stride=1, padding=2
- MaxPool1: 2x2, stride=2
- Conv2: 3x3 filters, 32 filters, stride=1, padding=1

**Questions:**
1. Output size after Conv1?
2. Output size after MaxPool1?
3. Output size after Conv2?
4. Total parameters in Conv1?
5. Total parameters in Conv2?

**Solution:**
```
1. After Conv1: [(32 - 5 + 4) / 1] + 1 = 32x32x16
2. After MaxPool1: [(32 - 2 + 0) / 2] + 1 = 16x16x16
3. After Conv2: [(16 - 3 + 2) / 1] + 1 = 16x16x32
4. Conv1 params: (5 × 5 × 3 + 1) × 16 = 76 × 16 = 1,216
5. Conv2 params: (3 × 3 × 16 + 1) × 32 = 145 × 32 = 4,640
```

---

## Problem Set 2: Decision Trees & Entropy

### Problem 2.1: Basic Entropy
**Given:** A dataset with the following class distribution:
- Class A: 8 samples
- Class B: 8 samples

**Question:** Calculate the entropy.

**Solution:**
```
P(A) = 8/16 = 0.5
P(B) = 8/16 = 0.5
E = -(0.5 × log2(0.5) + 0.5 × log2(0.5))
E = -(0.5 × (-1) + 0.5 × (-1))
E = -(-0.5 - 0.5) = 1.0
```

---

### Problem 2.2: Impure Split
**Given:**
- Total samples: 20 (12 Yes, 8 No)

After split on feature X:
- Left branch: 8 samples (7 Yes, 1 No)
- Right branch: 12 samples (5 Yes, 7 No)

**Questions:**
1. Calculate initial entropy
2. Calculate entropy of left branch
3. Calculate entropy of right branch
4. Which branch is purer?

**Solution:**
```
1. Initial: E = -(12/20 × log2(12/20) + 8/20 × log2(8/20))
           E = -(0.6 × log2(0.6) + 0.4 × log2(0.4))
           E = -(0.6 × (-0.737) + 0.4 × (-1.322))
           E = -(-0.442 - 0.529) = 0.971

2. Left: E_L = -(7/8 × log2(7/8) + 1/8 × log2(1/8))
             = -(0.875 × (-0.193) + 0.125 × (-3))
             = -(-0.169 - 0.375) = 0.544

3. Right: E_R = -(5/12 × log2(5/12) + 7/12 × log2(7/12))
              = -(0.417 × (-1.263) + 0.583 × (-0.779))
              = -(-0.527 - 0.454) = 0.981

4. Left branch is purer (lower entropy: 0.544 < 0.981)
```

---

### Problem 2.3: Perfect Split
**Given:** After a split:
- Branch A: All 10 samples are Class 1
- Branch B: 5 samples Class 2, 3 samples Class 3

**Questions:**
1. Entropy of Branch A?
2. Entropy of Branch B?

**Solution:**
```
1. Branch A: All same class → E = 0 (perfect purity)

2. Branch B: E = -(5/8 × log2(5/8) + 3/8 × log2(3/8))
             = -(0.625 × (-0.678) + 0.375 × (-1.415))
             = -(-0.424 - 0.531) = 0.955
```

---

## Problem Set 3: Support Vector Machines

### Problem 3.1: Decision Boundary
**Given:** SVM with decision function: `f(x) = 2x₁ + 3x₂ - 6`

**Questions:** Classify the following points:
1. (1, 2)
2. (3, 1)
3. (0, 2)
4. (4, -1)

**Solution:**
```
1. f(1,2) = 2(1) + 3(2) - 6 = 2 + 6 - 6 = 2 > 0 → POSITIVE
2. f(3,1) = 2(3) + 3(1) - 6 = 6 + 3 - 6 = 3 > 0 → POSITIVE
3. f(0,2) = 2(0) + 3(2) - 6 = 0 + 6 - 6 = 0 → ON BOUNDARY
4. f(4,-1) = 2(4) + 3(-1) - 6 = 8 - 3 - 6 = -1 < 0 → NEGATIVE
```

---

### Problem 3.2: Distance to Hyperplane
**Given:** `f(x) = x₁ - 2x₂ + 4`

**Question:** Which point is farthest from the decision boundary?
1. (2, 3)
2. (6, 1)
3. (0, 0)

**Solution:**
```
Distance ∝ |f(x)| / ||w|| where ||w|| = √(1² + (-2)²) = √5

1. |f(2,3)| = |2 - 2(3) + 4| = |2 - 6 + 4| = |0| = 0
2. |f(6,1)| = |6 - 2(1) + 4| = |6 - 2 + 4| = |8| = 8
3. |f(0,0)| = |0 - 0 + 4| = |4| = 4

Point (6,1) is farthest from the boundary.
```

---

## Problem Set 4: Neural Networks

### Problem 4.1: Forward Pass
**Given:**
- Input: x = 2
- Weight: w = 0.5
- Bias: b = 1
- Activation: Sigmoid

**Questions:**
1. Calculate z
2. Calculate ŷ using sigmoid
3. If true label y = 1, what's the loss? (Use squared loss)

**Solution:**
```
1. z = w × x + b = 0.5 × 2 + 1 = 2

2. ŷ = 1 / (1 + e⁻ᶻ) = 1 / (1 + e⁻²) 
     = 1 / (1 + 0.1353) = 1 / 1.1353 = 0.881

3. Loss = 0.5 × (ŷ - y)² = 0.5 × (0.881 - 1)²
        = 0.5 × (-0.119)² = 0.5 × 0.0142 = 0.0071
```

---

### Problem 4.2: Multiple Inputs
**Given:**
- Inputs: x = [1, 2, 3]
- Weights: w = [0.5, -0.3, 0.8]
- Bias: b = 0.5

**Question:** Calculate the linear output z.

**Solution:**
```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
z = 0.5(1) + (-0.3)(2) + 0.8(3) + 0.5
z = 0.5 - 0.6 + 2.4 + 0.5
z = 2.8
```

---

### Problem 4.3: Network Architecture
**Given:** A neural network with:
- Input layer: 10 neurons
- Hidden layer 1: 20 neurons (fully connected)
- Hidden layer 2: 15 neurons (fully connected)
- Output layer: 3 neurons (fully connected)

**Question:** How many total parameters (weights + biases)?

**Solution:**
```
Layer 1 → Hidden 1: (10 × 20) + 20 = 220
Hidden 1 → Hidden 2: (20 × 15) + 15 = 315
Hidden 2 → Output: (15 × 3) + 3 = 48

Total = 220 + 315 + 48 = 583 parameters
```

---

## Problem Set 5: Mixed Concepts

### Problem 5.1: CNN Step-by-Step
**Given:**
- Input matrix (4×4):
```
[1  2  3  4]
[5  6  7  8]
[9  10 11 12]
[13 14 15 16]
```
- Filter (2×2):
```
[1  0]
[0  1]
```
- Stride: 2, Padding: 0, Bias: 1

**Question:** Calculate the output feature map.

**Solution:**
```
Output size: [(4-2+0)/2] + 1 = 2×2

Position (0,0):
(1×1 + 2×0 + 5×0 + 6×1) + 1 = 1 + 6 + 1 = 8

Position (0,2):
(3×1 + 4×0 + 7×0 + 8×1) + 1 = 3 + 8 + 1 = 12

Position (2,0):
(9×1 + 10×0 + 13×0 + 14×1) + 1 = 9 + 14 + 1 = 24

Position (2,2):
(11×1 + 12×0 + 15×0 + 16×1) + 1 = 11 + 16 + 1 = 28

Output:
[8  12]
[24 28]
```

---

## Answer Key Summary

### CNN Dimensions
- Problem 1.1: 6×6, 160 params
- Problem 1.2: 5×5, 2,432 params
- Problem 1.3: 8×8
- Problem 1.4: See detailed solutions

### Entropy
- Problem 2.1: E = 1.0
- Problem 2.2: Left branch purer (E = 0.544)
- Problem 2.3: Branch A = 0, Branch B = 0.955

### SVM
- Problem 3.1: Positive, Positive, Boundary, Negative
- Problem 3.2: Point (6,1) farthest

### Neural Networks
- Problem 4.1: z=2, ŷ=0.881, Loss=0.0071
- Problem 4.2: z = 2.8
- Problem 4.3: 583 parameters

---

## Additional Practice

For more practice:
1. Create your own variations of these problems
2. Run the notebooks with different parameters
3. Try explaining concepts to a study partner
4. Draw network architectures and trace calculations

