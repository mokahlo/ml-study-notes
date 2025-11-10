# CEE 501 Midterm 2 Study Guide

## 📚 Study Materials Overview

### Available Resources
1. **cheat-sheet.txt** - Quick reference formulas and concepts
2. **CNN_MZA.ipynb** - Convolutional Neural Networks implementation
3. **Diabetes_ANN_MZA.ipynb** - Artificial Neural Networks implementation
4. **practice-problems.md** - Additional practice problems (this file)
5. **concept-review.md** - Detailed concept explanations

---

## 🎯 Key Topics Covered

### 1. Neural Network Fundamentals
- Linear calculations: `z = w * x + b`
- Activation functions (Sigmoid, ReLU, etc.)
- Loss functions (Squared Loss, Cross-Entropy)
- Backpropagation

### 2. Convolutional Neural Networks (CNNs)
- **Convolution operations**
- **Pooling layers**
- **Filter/Kernel dimensions**
- **Output size calculations**: `[(Input - Filter + 2*Padding) / Stride] + 1`
- **Parameter counting**: `(Filter_H × Filter_W × Depth + 1) × Num_Filters`

### 3. Support Vector Machines (SVMs)
- Decision boundaries
- Hyperplane equations
- Classification rules (positive/negative sides)
- Margin maximization

### 4. Decision Trees
- Entropy calculations: `E = -Σ(p_i * log2(p_i))`
- Information gain
- Splitting criteria
- Purity measures

### 5. Artificial Neural Networks (ANNs)
- Multi-layer perceptrons
- Training processes
- Evaluation metrics (accuracy, precision, recall)

---

## 📖 Study Plan

### Day 1-2: Review Fundamentals
- [ ] Review cheat sheet thoroughly
- [ ] Understand all formulas and when to apply them
- [ ] Run through CNN notebook
- [ ] Run through ANN/Diabetes notebook

### Day 3-4: Practice Problems
- [ ] Complete all practice problems
- [ ] Work through sample midterm questions
- [ ] Focus on calculation-heavy problems (CNN dimensions, entropy)

### Day 5: Review & Test
- [ ] Review incorrect practice problems
- [ ] Create summary notes
- [ ] Quick reference card for formulas

---

## 🔑 Quick Reference

### Most Important Formulas

#### CNN Output Size
```
Output = [(Input - Filter + 2*Padding) / Stride] + 1
```

#### CNN Parameters
```
Total = (FilterH × FilterW × InputDepth + 1) × NumFilters
```

#### Entropy
```
E = -Σ(p_i × log2(p_i))
```

#### SVM Classification
```
If f(x) > 0 → Positive class
If f(x) < 0 → Negative class
```

#### Neural Network Forward Pass
```
z = w * x + b
a = activation(z)
```

---

## 💡 Study Tips

1. **Understand, Don't Memorize**: Focus on understanding why formulas work
2. **Practice Calculations**: CNN dimensions and entropy require practice
3. **Run the Notebooks**: Execute all cells to see results
4. **Draw Diagrams**: Visualize CNN operations, decision boundaries
5. **Check Your Work**: Double-check stride and padding in CNN calculations

---

## 📝 Common Mistakes to Avoid

- ❌ Forgetting to add 1 in CNN output size formula
- ❌ Not counting bias terms in parameter calculations
- ❌ Mixing up stride vs. filter size
- ❌ Using natural log instead of log2 for entropy
- ❌ Confusing padding amount with padded size

---

## 🎓 Before the Exam

- [ ] Review all formulas on cheat sheet
- [ ] Practice at least 3 CNN dimension problems
- [ ] Practice at least 2 entropy calculations
- [ ] Understand SVM decision boundaries
- [ ] Know activation function properties

Good luck! 🍀
