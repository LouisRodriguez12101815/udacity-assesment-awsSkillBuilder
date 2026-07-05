---
layout: default
title: "Udacity Assessment Knowledge Base"
---

This page is a **concept-first study guide** (with deeper explanations) based on the questions we worked through.

## How to use this
- If you’re reviewing quickly: read the **Review** bullets.
- If something feels fuzzy: read the **Background** section right underneath.

---

# A) ML Optimization + Neural Nets

## 1) What is gradient descent for?
### Review
- **Minimize the cost/loss function** by iteratively updating model parameters.

### Background
Gradient descent updates parameters `theta` using the gradient of the loss `J(theta)`:
- `theta <- theta - alpha * grad_theta J(theta)`
`alpha` is the learning rate (step size). It’s an optimization method—**not** a method for choosing architecture (layers/activations).

## 2) How can gradient descent avoid getting stuck?
### Review
Helpful techniques include:
- **Stochastic gradients (SGD / mini-batches)**: noise can help escape shallow minima / saddle points.
- **Momentum** (and related optimizers): helps move through small traps/flat regions.
- **Random restarts**: different initializations, keep the best run.

Learning rate decay:
- Primarily helps **converge stably** once you’re near a good region.
- It can help as part of an annealing strategy, but it’s not the most direct “escape local minima” answer.

### Background
Deep nets often struggle more with **saddle points** and **plateaus** than “bad local minima.” Stochasticity and momentum help keep training moving.

## 3) What conditions are needed for gradient descent?
### Review
- The loss should be **differentiable** (so a gradient exists).
- Many beginner quizzes also say **continuous** (note: differentiable ⇒ continuous).
- “Minimized at discrete intervals” is **not** a requirement.

### Background
Gradient descent needs a direction to move, which comes from the derivative/gradient. In practice, ML often uses functions that are differentiable **almost everywhere** (e.g., ReLU).

## 4) What does an activation function do in a perceptron?
### Review
- It decides the neuron output from the **weighted sum + bias**.
- In a classic perceptron, this is often a **step/threshold** (fire vs not).

### Background
Perceptron:
- `z = w^T x + b` (weighted sum + bias)
- output `y_hat = phi(z)`
In deeper networks, activation functions introduce **nonlinearity**, enabling complex decision boundaries.

## 5) What does a high cross-entropy mean?
### Review
- The model assigns **low probability to the true class**.
- Often corresponds to predictions being **uncertain** or **confident-but-wrong**.

### Background
For one-hot classification, per example:
- `L = -log(p_true_class)`
If the model gives the correct class a small probability, loss becomes large.

---

# B) NLP + Transformers

## 6) What is tokenization for?
### Review
- Splits text into **tokens** (smaller units) that models can process.

### Background
Tokens can be words, characters, or **subwords** (common for modern LLMs). Tokenization is not compression, not speech-to-text, and not classification.

## 7) Which part is NOT in a Transformer encoder?
### Review
- **Masked** multi-head self-attention is **not** part of the encoder (it’s for the decoder).

### Background
The encoder can attend to all input tokens because the whole input is available. The decoder needs masking to generate left-to-right.

## 8) Why do we need positional encoding?
### Review
- To provide information about **token order / position**.

### Background
Self-attention alone doesn’t inherently encode order. Positional encodings inject order so the model can distinguish sequences with the same words in different arrangements.

## 9) How does masked attention prevent “cheating”?
### Review
- It blocks attention to **future tokens** (often by setting their attention logits to a very negative value, effectively −∞, so softmax weight becomes 0).

### Background
This is a **causal mask** (upper triangular). It enforces autoregressive generation: token at position `t` can only look at positions `<= t`.

## 10) What does the feedforward block do in a Transformer?
### Review
- Applies a **position-wise MLP** (Linear → nonlinearity → Linear) to each token representation.

### Background
The attention block mixes information across tokens. The feedforward block increases representational power **per token**, typically expanding then projecting back down (e.g., d_model → d_ff → d_model).

---

# C) Python fundamentals

## 11) How do you add a key-value pair to a dictionary?
### Review
- Use assignment: `d[key] = value`

### Background
- `append()` is for lists.
- `add()` is for sets.

## 12) What stores unique items and supports union/intersection?
### Review
- `set`

### Background
Sets enforce uniqueness and support `|` (union) and `&` (intersection).

## 13) What stores unique items where order isn’t guaranteed?
### Review
- `set`

### Background
Sets are unordered collections of unique hashable items.

## 14) How do you sort without modifying the original list?
### Review
- Use `sorted(my_list)` (returns a new list).

### Background
- `my_list.sort()` sorts in place and returns `None`.

## 15) How do you get the number of characters in a string?
### Review
- `len(s)`

---

# D) Visualization (Matplotlib)

## 16) Overplotting in scatter plots: how to reduce overlap?
### Review
- Use **transparency (`alpha`)** and **jitter**.

### Background
Transparency helps show density when points overlap. Jitter adds small random noise to separate points that share the same coordinates. Other common techniques include hexbin/2D histograms (when very dense).

## 17) What is `plt.subplot()` for?
### Review
- Creating/selecting **subplots (multiple axes)** inside a figure.

### Background
You use subplots to arrange multiple plots in a grid (e.g., 2 rows × 2 cols). Overlaying multiple plots on the same axes is usually just multiple plotting calls on the same axes.

## 18) Bar chart function?
### Review
- `plt.bar(...)`

## 19) Set x-axis limits?
### Review
- `plt.xlim(xmin, xmax)`

## 20) A key property of scatterplots?
### Review
- They visualize the relationship between **two numeric variables** (trend, correlation, clusters, outliers).

---

# Quick flash review (30 seconds)
- GD purpose → minimize loss by adjusting parameters
- Escape “stuck” → SGD noise, momentum, random restarts (LR decay mainly improves convergence)
- GD condition → differentiable loss
- Perceptron activation → decide output from weighted sum
- High cross-entropy → low prob on true class
- Tokenization → split into tokens
- Encoder ≠ masked self-attention
- Positional encoding → order/position info
- Masked attention → blocks future tokens
- Transformer FFN → position-wise MLP
- Dict add → `d[k]=v`
- Unique + set ops → set
- Sorted copy → `sorted()`
- String length → `len()`
- Overplotting → alpha + jitter
- subplot → multiple axes
- bar chart → `bar()`
- x-limits → `xlim()`
- scatterplot → relationship between two numeric vars
