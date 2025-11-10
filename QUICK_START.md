# 🚀 Quick Start Guide - Midterm 2 Study Center

## What You Now Have

Your Midterm 2 folder is now a **complete study center** with:
- ✅ 13 reference PDFs from Classes 8-12
- ✅ 3 Jupyter notebooks with implementations
- ✅ 3 comprehensive study guides
- ✅ 30+ practice problems with solutions
- ✅ Cheat sheet and sample exam

---

## ⏱️ 5-Minute Quick Start

### If you have LIMITED TIME:
1. **2 min**: Skim `STUDY_GUIDE.md` to see what topics are covered
2. **2 min**: Review `cheat-sheet.txt` for all formulas
3. **1 min**: Look at `MATERIALS_INDEX.md` topic mapping

### If you have 30 MINUTES:
1. Read `STUDY_GUIDE.md` completely
2. Work through 3-4 practice problems from `practice-problems.md`
3. Skim the relevant PDFs (start with `06_Intro_to_CNN.pdf`)

### If you have 1-2 HOURS:
1. Read `concept-review.md` thoroughly
2. Work through half of `practice-problems.md`
3. Run one of the Jupyter notebooks (`CNN_MZA.ipynb` or `Diabetes_ANN_MZA.ipynb`)
4. Review `cheat-sheet.txt`

### If you have 1+ DAYS (Ideal):
1. Follow the recommended study order in `MATERIALS_INDEX.md`
2. Read PDFs in this order: 
   - `02_Intro_Neural Network.pdf` → `03_Backpropagation.pdf`
   - `06_Intro_to_CNN.pdf` → `09_CNN_Advanced_Techniques.pdf`
3. Run all 3 Jupyter notebooks
4. Complete all practice problems
5. Take the sample midterm
6. Review weak areas

---

## 🎯 Files by Priority

### MUST READ (Highest Priority)
1. `STUDY_GUIDE.md` - Overview of all topics
2. `cheat-sheet.txt` - All formulas you need
3. `02_Intro_Neural Network.pdf` - NN fundamentals
4. `06_Intro_to_CNN.pdf` - CNN fundamentals

### SHOULD READ (High Priority)
5. `03_Backpropagation.pdf` - How NNs learn
6. `04_Training_Neural_Nets_Keras.pdf` - Training methods
7. `practice-problems.md` - Practice calculations
8. Sample midterm - Full exam simulation

### NICE TO READ (Medium Priority)
9. `concept-review.md` - Deep dive into concepts
10. `08_Network_Architectures.pdf` - Network designs
11. `09_CNN_Advanced_Techniques.pdf` - Advanced CNN

### OPTIONAL (Lower Priority)
12. `07_Transfer_Learning.pdf` - Advanced technique
13. `06_Training_Neural_Nets_PyTorch.pdf` - Framework specific
14. `10_Text_Word_Vectors.pdf` - NLP topics
15. `06-PCA.pdf` - Dimensionality reduction

---

## 💻 Running the Notebooks

### Setup
```powershell
# Navigate to folder
cd "g:\My Drive\CEE_501\Midterm 2"

# Open in VS Code
code .
```

### Recommended Execution Order
1. **CNN_MZA.ipynb** - See CNN in action
2. **Diabetes_ANN_MZA.ipynb** - See ANN in action
3. **PCA_MZA.ipynb** - See PCA in action

---

## 📝 Key Formulas (Most Important)

### CNN Output Size
```
Output = ⌊(Input - Filter + 2×Padding) / Stride⌋ + 1
```
**Remember the +1!**

### CNN Parameters
```
Params = (Filter_H × Filter_W × Input_Depth + 1) × Num_Filters
```

### Neural Network Forward Pass
```
z = w·x + b
a = activation(z)
Loss = -[y·log(a) + (1-y)·log(1-a)]  [Cross-entropy]
```

### Entropy (Decision Trees)
```
E = -Σ(p_i × log₂(p_i))
```
**Use log base 2, not natural log!**

### Gradient Descent
```
w_new = w_old - α × ∂L/∂w
```

---

## 🧠 Most Commonly Tested Topics

Based on class structure, expect questions on:

1. **CNN Dimension Calculations** (appears on every exam)
   - Output size with various stride/padding
   - Parameter counting

2. **Neural Network Concepts**
   - Activation functions and their properties
   - Backpropagation process
   - Loss functions

3. **Training Neural Networks**
   - Gradient descent variants
   - Learning rate effects
   - Overfitting vs underfitting

4. **Advanced Topics**
   - Transfer learning concepts
   - Network architecture choices
   - Batch normalization, dropout

---

## ❌ Common Mistakes to Avoid

1. **CNN Formulas**
   - ❌ Forgetting the +1 in output size
   - ❌ Not including bias in parameter count
   - ❌ Mixing stride with filter size

2. **Activation Functions**
   - ❌ Using wrong range expectations
   - ❌ Not knowing gradient properties

3. **Loss Functions**
   - ❌ Using ln instead of log₂ for entropy
   - ❌ Confusing squared loss with cross-entropy

4. **Study Approach**
   - ❌ Only memorizing without understanding
   - ❌ Not practicing calculations
   - ❌ Skipping the notebooks

---

## 📊 Checklist for Exam Day

- [ ] CNN output size formula memorized
- [ ] CNN parameter formula memorized
- [ ] Can calculate entropy by hand
- [ ] Know when to use each activation function
- [ ] Understand backpropagation concept
- [ ] Can explain overfitting and solutions
- [ ] Know key CNN architectures (VGG, ResNet, etc.)
- [ ] Understand transfer learning concept

---

## 🆘 If You Get Stuck

### Stuck on a Concept?
→ Check `concept-review.md` for detailed explanation

### Need Quick Formula?
→ Check `cheat-sheet.txt` for all formulas

### Need Practice?
→ Do problems in `practice-problems.md`

### Need to See Code?
→ Run the Jupyter notebooks in this folder

### Need Context?
→ Check `MATERIALS_INDEX.md` for which PDF covers what

---

## ⏰ Recommended Timeline

### 1 Week Before Exam
- [ ] Read all MUST READ files
- [ ] Work through all notebooks
- [ ] Complete 50% of practice problems

### 3 Days Before Exam
- [ ] Complete 100% of practice problems
- [ ] Review weak areas
- [ ] Read SHOULD READ files

### 1 Day Before Exam
- [ ] Take sample midterm timed
- [ ] Review incorrect answers
- [ ] Quick skim of concept-review.md
- [ ] Memorize formulas

### Exam Day Morning
- [ ] Quick review of cheat-sheet.txt
- [ ] Review common mistakes
- [ ] Breathe and be confident!

---

## 📞 Pro Tips

1. **Draw it out**: Visualize CNNs, networks, decision boundaries
2. **Teach it**: Explain concepts to a friend or out loud
3. **Vary practice**: Mix different problem types
4. **Track errors**: Note what types of problems you get wrong
5. **Active recall**: Test yourself without looking at notes
6. **Spacing**: Study over multiple days, not just one session

---

## 🎓 You're All Set!

You have everything needed to ace this midterm. The key is:
- **Understand** the concepts (not just memorize)
- **Practice** the calculations repeatedly
- **Review** your mistakes carefully

**Good luck! You've got this! 🍀**

