# Midterm High-Yield Practice Pack — CEE 501 (Condensed)

Purpose: If you have a limited time (tonight, tomorrow holiday, Wed evening, part of Thu evening, and a short lunch break on Fri), this packet contains the highest-yield problems and short solutions across the major topics likely on the midterm. Complete the problems in order and spend ~10–20 minutes on each, except the two "longer" (multi-part) questions.

How to use
- Night 1 (tonight): items 1–7
- Holiday day: items 8–15 (deeper practice; do the long ones)
- Wed evening: review mistakes + redo 3 weak problems
- Thu evening (short): quick review of answers + practice test
- Fri lunch: timed 30–45 minute 5-question run (pick from 1–15)

Topics covered (quick list): CNN dims & params, convolution arithmetic, pooling, filter computation, NN forward pass & loss, backprop intuition, entropy/info gain, SVM boundary & margin, PCA/std, transformer attention, optimization & hyperparameters, regularization.

---

## Quick pack: 18 problems (high yield)

1) CNN Parameter & Output (Short)
Given a 28×28 grayscale image, conv layer with 3×3 filters, stride=1, padding=1, 16 filters. What's output shape and how many parameters?

Solution (short): Output = 28×28×16. Parameters = (3×3×1 + 1) × 16 = 10×16 = 160.

2) Convolution with Stride and Padding
Input 32×32×3, filter 5×5, stride=2, padding=2, 32 filters. Output shape?

Solution: Each dimension: floor((32 -5+2*2)/2)+1 = floor((32 -5+4)/2)+1 = floor(31/2)+1 = 15+1=16. Output 16×16×32.

3) Pooling
Input 16×16 feature map, 2×2 max pool, stride=2. Output size?

Solution: Output = [(16-2)/2] +1 = 8×8.

4) CNN filter computation (4×4 example)
Given X= [[1,2,3,0],[0,1,2,3],[3,1,0,2],[2,0,1,1]], K=[[1,0],[-1,1]], stride=2, bias=1. Compute output. (This appears in the practice set.)

Solution: 2×2; out = [[3,5],[2,1]] (walk through 4 patches).

5) Basic Forward Pass (NN)
x=2, w=0.5, b=1, sigmoid activation. Compute z, ŷ, squared loss if y=1.

Solution: z=2; ŷ≈0.881; Loss=0.5*(0.881-1)^2 ≈ 0.0071.

6) Multiple-input linear layer
x=[1,2,3], w=[0.5,-0.3,0.8], b=0.5. Compute z.

Solution: z = 0.5 -0.6 +2.4 +0.5 = 2.8.

7) Backprop intuition (short)
For the neuron in #5, compute derivative of squared loss w.r.t. weight w (numerical). y=1.

Solution: (ŷ - y)*ŷ*(1-ŷ)*x. Compute numerically: ŷ=0.881, ŷ(1-ŷ)=0.881*0.119≈0.105, (ŷ-y)=-0.119; grad = -0.119 *0.105 *2 ≈ -0.025.

8) Entropy (decision trees)
Yes=6, No=2. Compute H (base 2).

Solution: p1=0.75, p2=0.25. H=-0.75 log2 0.75 -0.25 log2 0.25 ≈ 0.811.

9) Entropy gain (practical)
Given parent: 20 samples (12 Yes, 8 No). After split: left 8 (7Y,1N), right 12 (5Y,7N). Compute information gain; identify better/worse branch.

Solution: Initial E=0.971. Left 0.544 Right 0.981. Weighted avg = (8/20)*0.544 + (12/20)*0.981 = 0.824. Gain=0.971-0.824=0.147.

10) SVM classification & margin
Given f(x)=2x1+3x2-6. Determine classification for (1,2),(3,1),(0,2),(4,-1) and which is farthest from hyperplane.

Solution: Evaluate f: (1,2)=2>0 pos, (3,1)=3 pos, (0,2)=0 boundary, (4,-1)= -1 neg. Dist ∝ |f(x)|; compute magnitudes: 2,3,0,1 → (3,1) farthest.

11) PCA & feature scaling short
Which features must you scale? Why?

Solution: Any features with widely different numeric ranges must be standardized before PCA. Otherwise the high magnitude features dominate variance.

12) Transformer attention conceptual (short)
What is attention for? Role of FFN?

Solution: Attention creates token-wise weighted sum of other token embeddings; FFN refines transformed token embedding with nonlinear projection.

13) CNN parameter example (deeper)
Input 32×32×3; Conv1: 5×5 filters 16 stride=1 padding=2; MaxPool: 2×2 stride=2; Conv2: 3×3 filters 32 stride=1 padding=1. Compute after conv1, after pool, after conv2, and params.

Solution: After Conv1 32×32×16, pool 16×16×16, conv2 16×16×32. Params Conv1 (5*5*3+1)*16=76*16=1216. Conv2 (3*3*16+1)*32 = (144+1)*32 =145*32=4640.

14) CNN conv output with stride/padding (deeper)
Given 10×10 and 5×5 filter stride=2 padding=2, compute output and params with 32 filters and input depth 3.

Solution: ((10-5+4)/2)+1 = ((9)/2)+1 = 4+1=5 → 5×5. Params = (5*5*3+1)*32 = (75+1)*32=76*32=2432.

15) A longer combined problem (multi-part: conv dims, one filter calc, forward pass)
- Part A: Given 28×28×1 conv 3×3 stride=1 padding=1 with 10 filters. Output dims and params.
- Part B: Using the output feature map at position (0,0), compute conv result for a provided patch and filter.
- Part C: Using one result, compute loss w/ given target.

Solution (sketch): A: 28×28×10; params = (9*1+1)*10=10*10=100. B: walk through patch multiply-add plus bias. C: compute error.

16) Optimization & learning rate (short)
Explain effect of too-small and too-large learning rates; illustrate with one step of gradient descent on f(x)= (x-3)^2 starting from x0=0 and lr=0.1 vs lr=1.

Solution: gradient = 2(x-3); from x0=0 grad=-6. Step: lr=0.1 → x1 = 0 -0.1*(-6) = 0.6 (closer). lr=1 → x1=0 -1*(-6)=6 overshoot (way past). Too large → may diverge.

17) Regularization (short)
Explain L2 vs dropout usage and their effect on overfitting.

Solution: L2 penalizes weights encourages smaller coefficients; dropout randomly removes activations simulating ensemble. Both reduce overfitting—L2 by constraining weights, dropout by reducing co-adaptations.

18) Exam-style: Short 5-question timed test (pick 5—mix across above)
Pick 5 questions (mix CNN dims, conv computation, forward pass, entropy, SVM) and time 30–45 minutes.

---

## Study Plan (condensed, suggested)
Assumptions: total time ≈ 10-12 hours across the period. Adjust durations if needed.

Night 1 (2–3 hrs): Start with quick items 1–7. Work each problem, check solutions, and make short notes on mistakes.

Day (Holiday) (3–4 hrs): Do problems 8–15. Spend more time on 13, 15 (deeper), and do the combined problem step-by-step. At the end, rework 3 problems you found hard.

Wed evening (2 hrs): Re-do the 5 problems you missed. Work on backprop/derivative practice (problem 7 and similar ones from other sets). Quick review notes.

Thu evening (1.5–2 hrs): Timed practice test (pick questions 18 and 5 others), then review wrong answers.

Fri lunch (30–45 min): Final timed 30–45 minute test from the shorter problems (1–6 & 8 & 10). Quick review.

---

## Quick Tools & Next Steps
- Add a simple notebook that auto-checks numeric results for items 4,5,6,7,8,13,15 & 18. (Ask if you want this.)
- If any specific topics need more attention, tell me (e.g., backprop algebra, cross-entropy, PCA proof) and I’ll add a short micro-lesson + 3 practice items.

Good luck — tell me if you want the interactive notebook auto-checker next, or a printable two-page worksheet for last-minute review.
