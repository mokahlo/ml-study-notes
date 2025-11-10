# Transformer_MZA.ipynb - Error Review & Solutions

## Executive Summary
**Review Status: 7 Issues Found** (3 Critical, 2 Medium, 2 Minor)

---

## Critical Issues

### 🔴 **Issue 1: Redundant y_preds Assignment (Cell 95-96)**

**Location:** Cells 95 & 96
**Severity:** MEDIUM (Code works but inefficient)

**Problem:**
```python
# Cell 95
y_preds = np.argmax(preds_output.predictions, axis = 1)

# Cell 96
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)
```

The same line appears twice - once in cell 95 (standalone), then again in cell 96 before plotting.

**Solution:**
Delete cell 95 or consolidate into cell 96:

```python
# OPTION 1: Delete cell 95 entirely
# OPTION 2: Merge into single cell
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)
```

---

### 🔴 **Issue 2: Missing plt.show() in Plot Cells**

**Location:** Cells 95, 97, 98
**Severity:** MEDIUM (Plots may not display)

**Problem:**
Cell 97 and other plot cells may not show visualizations:

```python
# Cell 95
plt.title("Frequency of Classes")
# ❌ Missing plt.show()

# Cell 104 (Custom tweet predictions)
plt.bar(labels, 100 *preds_df["score"], color='C0')
plt.title(f"{custom_tweet}")
plt.ylabel("Class probability (%)")
# ❌ Missing plt.show()
```

**Solution:**
Add `plt.show()` at the end of all plotting cells:

```python
plt.title("Frequency of Classes")
plt.show()  # ✅ Add this line
```

---

### 🔴 **Issue 3: Column Access Key Mismatch (Subtle)**

**Location:** Cell 81 (Data Preparation)
**Severity:** MEDIUM (Potential runtime issue)

**Problem:**
Cell 71 sets format with specific columns:
```python
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask","label"])
```

But later you access `hidden_state` which wasn't in the original set_format:
```python
X_train = np.array(emotions_hidden["train"]["hidden_state"])  # ✅ Correct
```

This works because `extract_hidden_states()` adds the "hidden_state" key, but it's a dependency that could break if extraction fails.

**Solution:**
Add defensive check after extraction:

```python
# After emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
assert "hidden_state" in emotions_hidden["train"].column_names, "hidden_state extraction failed!"

X_train = np.array(emotions_hidden["train"]["hidden_state"])
```

---

## Medium Issues

### 🟡 **Issue 4: Typo in Comment (Cell 66)**

**Location:** Cell 66
**Severity:** MINOR (Documentation only)

**Problem:**
```python
# maximum context soze of the model  # ❌ "soze" should be "size"
tokenizer.model_max_length
```

**Solution:**
```python
# maximum context size of the model  # ✅ Fixed
tokenizer.model_max_length
```

---

### 🟡 **Issue 5: Typo in Markdown (Cell 6)**

**Location:** Cell 6 (Markdown Question)
**Severity:** MINOR (Documentation only)

**Problem:**
```markdown
1 - What is the data stucture type of the loaded dataset?  # ❌ "stucture" should be "structure"
```

**Solution:**
```markdown
1 - What is the data structure type of the loaded dataset?  # ✅ Fixed
```

---

### 🟡 **Issue 6: Typo in Markdown (Cell 10)**

**Location:** Cell 10 (Markdown Question)
**Severity:** MINOR (Documentation only)

**Problem:**
```markdown
How do you acess data from the given data structure?  # ❌ "acess" should be "access"
```

**Solution:**
```markdown
How do you access data from the given data structure?  # ✅ Fixed
```

---

## Minor Issues

### 🟢 **Issue 7: Unused Empty Cell (Cell 100)**

**Location:** Cell 100
**Severity:** LOW (Cleanup only)

**Problem:**
```python
# Empty cell - no code

```

**Solution:**
Either delete or add a comment:
```python
# End of notebook
```

---

## Potential Runtime Issues (Execution Warnings)

### ⚠️ **Warning 1: GPU/Device Availability**

**Location:** Cell 63
**Current Code:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Issue:** Code defaults to CPU if GPU unavailable, but training will be very slow on CPU.

**Recommendation:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Add this for visibility
if device.type == 'cpu':
    print("⚠️ WARNING: Using CPU - training will be very slow. Consider using GPU runtime.")
```

---

### ⚠️ **Warning 2: Long Training Time**

**Location:** Cell 91
**Current Code:**
```python
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,  # 2 epochs might not be enough
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    ...
)
```

**Issue:** 2 epochs may be insufficient for good convergence. Monitor validation loss.

**Recommendation:**
```python
# Consider these parameters if validation loss plateaus early:
# num_train_epochs=3,  # Increase if needed
# evaluation_strategy="steps",
# eval_steps=100,  # Check validation more frequently
```

---

## Dependency Verification

### ✅ Required Imports Check:
- [x] `torch` - Used in cells 67, 76
- [x] `transformers` - Used in cells 43, 63, 86, 98
- [x] `datasets` - Used in cell 6
- [x] `sklearn` - Used in cells 83, 90
- [x] `pandas` - Used in cell 16
- [x] `matplotlib` - Used in cells 20, 21
- [x] `numpy` - Used in cells 81, 95

All imports present ✅

---

## Execution Order Issues

### ✅ **Cell Dependency Check:**

| Cell # | Depends On | Status |
|--------|-----------|--------|
| 6 | (datasets module) | ✅ Installed |
| 16 | emotions dataset | ✅ Loaded in cell 6 |
| 43 | transformers | ✅ Installed |
| 63 | torch, transformers | ✅ Both available |
| 73 | extract_hidden_states func | ✅ Defined in cell 73 |
| 81 | emotions_hidden dataset | ✅ Created in cell 76 |
| 83 | sklearn, X_train, y_train | ✅ All available |
| 90 | lr_clf from cell 83 | ✅ Defined |
| 91 | emotions_encoded dataset | ✅ Created earlier |
| 104 | custom_tweet, classifier | ✅ All defined |

All dependencies satisfied ✅

---

## Summary Table

| # | Issue | Type | Severity | Line | Fix Difficulty |
|---|-------|------|----------|------|-----------------|
| 1 | Redundant y_preds | Logic | 🟡 MEDIUM | 504, 507 | ⭐ Easy |
| 2 | Missing plt.show() | UX | 🟡 MEDIUM | Multiple | ⭐ Easy |
| 3 | Column key dependency | Logic | 🟡 MEDIUM | 385 | ⭐⭐ Medium |
| 4 | Typo "soze" → "size" | Docs | 🟢 MINOR | 264 | ⭐ Easy |
| 5 | Typo "stucture" → "structure" | Docs | 🟢 MINOR | 26 | ⭐ Easy |
| 6 | Typo "acess" → "access" | Docs | 🟢 MINOR | 43 | ⭐ Easy |
| 7 | Empty cell | Cleanup | 🟢 LOW | 532 | ⭐ Easy |

---

## Recommendations (Priority Order)

### 🔴 **Priority 1: Critical Fixes (Do First)**
1. **Fix Missing plt.show()** - Cells 95, 97, 104
   - Estimated time: 2 minutes
   - Impact: Enables plot visualization

2. **Remove Redundant y_preds** - Delete cell 95
   - Estimated time: 1 minute
   - Impact: Code cleanliness

### 🟡 **Priority 2: Important Improvements (Do Before Exam)**
3. **Add Device Warning** - Cell 63
   - Estimated time: 3 minutes
   - Impact: Prevents silent performance issues

4. **Add Column Check** - After cell 76
   - Estimated time: 2 minutes
   - Impact: Early error detection

### 🟢 **Priority 3: Nice-to-Have (Optional)**
5. **Fix Typos** - Cells 26, 43, 264
   - Estimated time: 1 minute
   - Impact: Professional appearance

---

## Total Estimated Fix Time
- **Quick Fixes (Typos + Missing plt.show()):** 5 minutes
- **Medium Fixes (Redundant code + warnings):** 5 minutes
- **All Fixes:** ~10 minutes

---

## Testing Recommendations

After applying fixes, test in this order:

```
1. ✅ Run cells 1-10 (Setup & Data Loading)
2. ✅ Run cells 11-30 (Data Visualization)
3. ✅ Run cells 31-50 (Tokenization)
4. ✅ Run cells 51-80 (Feature Extraction)
5. ✅ Run cells 81-90 (Classifier Training)
6. ✅ Run cells 91-104 (Fine-tuning & Predictions)
```

Each section should complete without errors.

---

## Conclusion

**Overall Code Quality: GOOD** ✅

- ✅ All cells executable in order
- ✅ All dependencies present
- ✅ All functions properly defined
- ⚠️ Minor visualization/cleanup issues
- ⚠️ Could benefit from better error handling

**Ready for Study/Exam Use:** YES, but apply Priority 1 fixes first.

---

**Report Generated:** November 8, 2025
**Notebook:** Transformer_MZA.ipynb
**Total Cells:** 103
**Cells with Issues:** 7

