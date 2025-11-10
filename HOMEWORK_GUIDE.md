# 📝 Homework Materials for Midterm 2 Study

**Status**: ✅ HW3, HW4, and HW5 materials added to study center

---

## 📚 What's Included from Homeworks

### HW_2: Linear Regression & Gradient Descent
**Relevance**: ⭐⭐⭐ HIGH (Foundational for neural networks)

**File**: `HW-2_Linear-Reg-and-Gradient-Descent_MZA.ipynb`

**Topics Covered**:
- Linear regression basics
- Gradient descent algorithm
- Learning rate effects
- Convergence analysis

**Why Important for Midterm**:
- Gradient descent is core to NN training
- Helps understand optimization concepts
- Shows practical implementation

---

### HW_3: ML Classifiers
**Relevance**: ⭐⭐ MEDIUM (Context for neural networks)

**Files**:
- `HW3 - ML Classifiers.pdf`
- `CEE501_HW3_ML_Classifiers_MZA.ipynb`
- `CEE501_HW3_ML-Classifiers_MZA.ipynb - Colab.pdf`

**Topics Covered**:
- Support Vector Machines (SVM)
- Decision Trees
- K-Nearest Neighbors (KNN)
- Random Forests
- Naive Bayes
- Model comparison

**Why Important for Midterm**:
- Questions may compare neural networks to other classifiers
- Understanding SVM decision boundaries (likely on midterm)
- Tree-based algorithms understanding
- Classification evaluation metrics

---

### HW_4: Neural Networks with Keras
**Relevance**: ⭐⭐⭐⭐ VERY HIGH (Direct midterm content)

**Files**:
- `HW4.pdf` - Problem statement
- `concrete_data_NN_template.ipynb` - Starting template
- `1_concrete_data_NN_MZA.ipynb` - Solution (Concrete dataset)
- `2_annealing_data_NN_MZA.ipynb` - Solution (Annealing dataset)

**Topics Covered**:
- Neural network architecture design
- Layer specifications
- Activation functions in practice
- Model training and tuning
- Hyperparameter optimization
- Model evaluation

**Why Important for Midterm**:
- ✅ **MOST RELEVANT** - directly covers NN implementation
- Shows how to build networks in code
- Demonstrates parameter calculations
- Shows training process step-by-step

---

### HW_5: Traffic Sign Classification with CNNs
**Relevance**: ⭐⭐⭐⭐ VERY HIGH (CNN exam content)

**Files**:
- `HW5-1.pdf` - Problem statement
- `HW5_Traffic_Sign_Classification_MZA.ipynb` - Complete solution

**Topics Covered**:
- CNN architecture design
- Image preprocessing
- Convolutional layers
- Pooling layers
- Feature maps
- Training CNNs
- Transfer learning basics

**Why Important for Midterm**:
- ✅ **MOST CRITICAL** - CNN is major exam topic
- Shows real CNN implementation
- Demonstrates output size calculations
- Shows how to count parameters in practice
- Includes transfer learning example

---

## 🎯 Recommended HW Study Order

### For CNN Understanding (Highest Priority)
1. **Start**: `HW5_Traffic_Sign_Classification_MZA.ipynb`
   - See CNN in action on real image data
   - Understand layer-by-layer structure
   - Learn practical CNN design
   - ~2 hours to review

2. **Then**: `HW4.pdf` + `1_concrete_data_NN_MZA.ipynb`
   - See neural networks in practice
   - Understand model compilation and training
   - Learn hyperparameter tuning
   - ~1.5 hours to review

3. **Support**: `HW-2_Linear-Reg-and-Gradient-Descent_MZA.ipynb`
   - Review gradient descent fundamentals
   - See optimization in practice
   - ~1 hour to review

### For Understanding Different Classifiers
4. **Optional**: `HW3 - ML Classifiers.pdf` + notebook
   - Compare NN to SVM, trees, etc.
   - Understand classification concepts
   - ~1.5 hours to review

---

## 📊 Homework Topic Coverage vs Midterm Topics

| Midterm Topic | Related Homework | Study Priority |
|---------------|------------------|-----------------|
| **Forward Pass** | HW4, HW2 | ⭐⭐⭐⭐ Very High |
| **Backpropagation** | HW4, HW2 | ⭐⭐⭐⭐ Very High |
| **Gradient Descent** | HW2 | ⭐⭐⭐⭐ Very High |
| **Neural Networks** | HW4 | ⭐⭐⭐⭐ Very High |
| **CNN Architecture** | HW5 | ⭐⭐⭐⭐ Very High |
| **CNN Dimensions** | HW5 | ⭐⭐⭐⭐⭐ CRITICAL |
| **Parameter Counting** | HW5 | ⭐⭐⭐⭐⭐ CRITICAL |
| **Training Methods** | HW4 | ⭐⭐⭐ High |
| **Activation Functions** | HW4 | ⭐⭐⭐ High |
| **Loss Functions** | HW4 | ⭐⭐⭐ High |
| **Transfer Learning** | HW5 | ⭐⭐⭐ High |
| **SVM** | HW3 | ⭐⭐ Medium |
| **Decision Trees** | HW3 | ⭐⭐ Medium |

---

## 💡 How to Use the HW Materials

### Method 1: Learn by Example (Recommended)
1. Open `HW5_Traffic_Sign_Classification_MZA.ipynb`
2. Read through the implementation
3. Run each cell and see outputs
4. Match code concepts to class PDFs
5. Understand how theory becomes practice
6. Modify parameters and observe changes

### Method 2: Practice from Scratch
1. Open `HW5-1.pdf` (the problem statement)
2. Try to build your own CNN solution
3. Compare your code to the notebook solution
4. Identify what you did differently
5. Understand why they made certain choices

### Method 3: Fill in the Blanks
1. Open `concrete_data_NN_template.ipynb` (HW4)
2. Try to complete the template
3. Check your work against `1_concrete_data_NN_MZA.ipynb`
4. Learn the standard way to build NNs in Keras

### Method 4: Quick Reference
- Use HW notebooks as code examples when reviewing
- When you see a concept in class PDFs, find the HW example
- See how theory is implemented in practice

---

## 🔑 Critical Concepts from Homeworks

### From HW5 (CNN - Most Critical)
- **Conv2D layers**: How to specify filters, kernel size, stride, padding
- **Output dimensions**: Calculated in practice - verify your math
- **Pooling**: MaxPooling2D reduces dimensions
- **Flattening**: Transition from conv to dense layers
- **Parameter counting**: See total params in model summary
- **Training**: Epochs, batch size, optimization

### From HW4 (Neural Networks)
- **Sequential model**: How to stack layers
- **Dense layers**: Fully connected layers with weights/biases
- **Activation functions**: relu, sigmoid, softmax in practice
- **Compilation**: loss function, optimizer, metrics
- **Fitting**: Training with validation split
- **Evaluation**: Testing on unseen data
- **Hyperparameter tuning**: Changing architecture to improve performance

### From HW2 (Gradient Descent)
- **Learning rate**: How it affects convergence
- **Cost function**: How it changes over iterations
- **Batch updates**: Weight adjustments per batch
- **Convergence**: When to stop training

### From HW3 (ML Classifiers)
- **SVM decision boundaries**: Separating hyperplanes
- **Classification metrics**: Precision, recall, F1
- **Model comparison**: Which classifier works best
- **Cross-validation**: Evaluating model performance

---

## 📋 HW Files Now in Midterm 2

```
HW Materials Added:
├── HW-2_Linear-Reg-and-Gradient-Descent_MZA.ipynb
├── HW3 - ML Classifiers.pdf
├── CEE501_HW3_ML_Classifiers_MZA.ipynb
├── CEE501_HW3_ML-Classifiers_MZA.ipynb - Colab.pdf
├── HW4.pdf
├── concrete_data_NN_template.ipynb
├── 1_concrete_data_NN_MZA.ipynb
├── 2_annealing_data_NN_MZA.ipynb
├── HW5-1.pdf
└── HW5_Traffic_Sign_Classification_MZA.ipynb
```

---

## ⏱️ Time Allocation for HW Study

### If you have 2-3 hours:
1. **HW5 CNN notebook**: 90 minutes
   - This is THE most critical for midterm
   - See CNN implemented end-to-end
2. **HW4 NN notebooks**: 60 minutes
   - See NN training in practice

### If you have 4-5 hours:
1. **HW5 CNN**: 90 minutes
2. **HW4 NN**: 75 minutes
3. **HW2 Gradient Descent**: 60 minutes
   - Understand optimization
4. **Review practice problems**: 30 minutes

### If you have 1+ days:
1. Try to solve HW problems from scratch first
2. Compare your solutions to provided solutions
3. Identify gaps in understanding
4. Review related class PDFs
5. Run notebooks with modifications
6. Create your own variations

---

## 🎯 What Each HW Teaches About The Midterm

### HW2 Teaches You:
- How gradient descent actually works in code
- How learning rate affects training
- What the loss curve looks like during training

### HW3 Teaches You:
- How different classifiers make decisions
- What SVM decision boundaries look like
- How to evaluate classifier performance

### HW4 Teaches You:
- How to specify NN architecture in Keras
- How layers connect (Sequential model)
- What activation functions look like in practice
- How to count total parameters in a model
- How training process works (epochs, batches)

### HW5 Teaches You:
- How to build CNN architecture
- How Conv2D layer is implemented
- How output dimensions are calculated in practice
- How to count parameters in convolutional layers
- How to work with image data
- How to use transfer learning with CNNs

---

## ✅ Pre-Exam HW Checklist

- [ ] Run HW5_Traffic_Sign_Classification_MZA.ipynb
- [ ] Understand each layer in the CNN
- [ ] Verify parameter calculations match theory
- [ ] Run HW4 (at least one concrete NN notebook)
- [ ] See neural network training in action
- [ ] Run HW2 gradient descent notebook
- [ ] Understand learning rate effects
- [ ] Review HW3 for SVM and classifier concepts
- [ ] Try to build one model from scratch
- [ ] Compare your code to solution notebooks

---

## 🚀 How This Strengthens Your Study

**Homework Advantages**:
- ✅ See theory applied to real problems
- ✅ Understand library functions (Keras, PyTorch)
- ✅ Practice parameter calculations in context
- ✅ Learn debugging and troubleshooting
- ✅ Understand practical constraints
- ✅ See multiple solutions to same problem
- ✅ Get comfortable with code structure

**Your Study Center Now Has**:
- Class lecture materials (understanding)
- Homework examples (practice & application)
- Practice problems (test yourself)
- Study guides (organization)
- Sample exam (full simulation)

This is a **complete preparation package**! 🎓

