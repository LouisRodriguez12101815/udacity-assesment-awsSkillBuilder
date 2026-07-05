---
layout: default
title: "Udacity Assessment Knowledge Base"
---

This site is a **concept-first study guide** for common topics that show up in the Udacity AWS AI Programmer assessment.
It’s written for **review + understanding**, not as a list of Q&A.

## Key concepts & definitions (fast lookup)
### Machine learning
- **Loss / cost function**: a number that measures how wrong the model is; training tries to *minimize* it.
- **Gradient descent**: iteratively updates parameters to reduce loss by stepping in the negative gradient direction.
- **Learning rate (`alpha`)**: step size for parameter updates.
- **Momentum**: uses a running “velocity” of gradients to smooth updates and speed progress.
- **Cross-entropy**: classification loss; large when the model assigns low probability to the true class.
- **Activation function**: maps weighted sums to outputs; adds nonlinearity (or thresholding in a perceptron).

### NLP & Transformers
- **Tokenization**: splitting text into tokens (words/subwords/chars) that can be converted to IDs.
- **Self-attention**: lets each token attend to other tokens to build context-aware representations.
- **Positional encoding**: adds token order/position information (attention alone doesn’t encode order).
- **Causal (masked) attention**: blocks attention to future tokens to prevent “cheating” during generation.
- **Feedforward network (FFN)**: position-wise MLP applied to each token after attention.

### Python
- **Dictionary (`dict`)**: key → value mapping. Add/update via `d[key] = value`.
- **Set (`set`)**: unordered collection of unique items; supports union/intersection.
- **`sorted()` vs `.sort()`**: `sorted(x)` returns a new list; `list.sort()` mutates in place.
- **`len()`**: returns the length of a sequence (string length = number of characters).

### Matplotlib
- **Scatter plot**: shows relationship between two numeric variables.
- **Overplotting**: too many points overlap; mitigate with `alpha` (transparency) and jitter.
- **Subplots**: multiple axes in one figure (`plt.subplot(...)`).
- **Bar chart**: `plt.bar(...)`.
- **Axis limits**: `plt.xlim(xmin, xmax)`.

---

# Machine learning (concepts)
## Gradient descent (purpose & update rule)
### Review
- Gradient descent is used to **minimize a loss function** by adjusting model parameters.

### Background
Given parameters `theta` and loss `J(theta)`:
- `theta <- theta - alpha * grad_theta J(theta)`
`alpha` is the learning rate (step size). Gradient descent is an optimization method; it does **not** choose architecture (layers, activations).

## Learning rate schedules (including decay)
### Review
- A high learning rate can speed early learning but risk instability.
- **Learning rate decay** helps **stabilize convergence** later in training.

### Background
Many training setups start with a larger step size (or noisier updates) and reduce it over time (decay/annealing). This is mainly about *converging smoothly*, not the most direct “escape local minima” tool.

## Escaping poor basins (local minima / saddle points)
### Review
Useful techniques include:
- **SGD / mini-batches**: gradient noise can help escape shallow basins and saddle points.
- **Momentum / Adam-like optimizers**: can push through small traps and speed across plateaus.
- **Random restarts**: try multiple initializations, keep the best.

### Background
In deep learning, “getting stuck” is often about **saddle points** and **flat regions** rather than classic bad local minima. Noise and momentum keep optimization moving.

## Differentiability (what gradient methods require)
### Review
- Classic gradient descent assumes the loss is **differentiable** (so a gradient exists).
- Many quizzes also say **continuous** (note: differentiable ⇒ continuous).

### Background
In practice, ML often uses functions that are differentiable **almost everywhere** (e.g., ReLU), and training uses subgradients or practical approximations.

## Activation functions (perceptron intuition)
### Review
- The activation function decides a neuron’s output from the **weighted sum + bias**.
- A classic perceptron often uses a **threshold/step** (fire vs not).

### Background
Perceptron:
- `z = w^T x + b`
- output `y_hat = phi(z)`
In deeper networks, activations introduce **nonlinearity**, enabling complex decision boundaries.

## Cross-entropy (interpretation)
### Review
- High cross-entropy means the model assigns **low probability to the true class**.

### Background
For one-hot classification:
- `L = -log(p_true_class)`
If `p_true_class` is small, the loss becomes large. Dataset noise can be a *cause* of high loss, but the loss value itself measures prediction-vs-label mismatch.

---

# NLP & Transformers (concepts)
## Tokenization
### Review
- Tokenization splits text into **tokens** that models can process.

### Background
Tokens can be words, characters, or subwords (common for modern LLMs). Tokenization is not compression, speech-to-text, or classification.

## Encoder vs decoder attention (masked vs unmasked)
### Review
- **Encoder** self-attention is typically **unmasked** (can attend to all input tokens).
- **Decoder** uses **masked (causal)** self-attention to avoid looking at future tokens.

### Background
The encoder sees the entire input sequence. The decoder generates left-to-right, so it must not “peek” ahead.

## Positional encoding
### Review
- Positional encodings provide information about **token order / position**.

### Background
Self-attention by itself doesn’t encode order; positional signals inject the missing sequence structure.

## Causal masking (“prevent cheating”)
### Review
- Masked attention blocks future tokens by zeroing out their attention weights (often by setting logits to a very negative value, effectively −∞).

### Background
This is a causal (upper-triangular) mask: a token at position `t` can only attend to positions `<= t`.

## Feedforward block (FFN)
### Review
- The Transformer FFN is a **position-wise MLP** (Linear → nonlinearity → Linear) applied to each token.

### Background
Attention mixes information across tokens; the FFN increases representational power **per token**, often via an expand-and-project pattern (`d_model -> d_ff -> d_model`).

---

# Python (language fundamentals)
## Dictionaries (`dict`)
### Review
- Add/update: `d[key] = value`.

### Background
- `append()` is for lists.
- `add()` is for sets.

## Sets (`set`)
### Review
- Store **unique** items and support union/intersection.

### Background
Sets are unordered and support:
- union: `A | B`
- intersection: `A & B`

## Sorting (copy vs in-place)
### Review
- `sorted(x)` returns a new list.
- `list.sort()` mutates the list and returns `None`.

## String length
### Review
- Use `len(s)`.

---

# Matplotlib (tooling)
## Scatter plots & overplotting
### Review
- Scatter plots show the relationship between two numeric variables.
- Reduce overlap with **transparency** (`alpha`) and **jitter**.

### Background
`alpha` makes dense regions visible. Jitter separates points that share identical coordinates. For extreme density, hexbin/2D histograms are also common.

## Subplots (`plt.subplot`)
### Review
- `plt.subplot(...)` creates/selects axes in a grid to show multiple plots in one figure.

### Background
Overlaying multiple plots on the same axes is typically just multiple plot calls on the same axes; `subplot` is for *multiple axes*.

## Bar charts
### Review
- Use `plt.bar(...)`.

## Axis limits
### Review
- Use `plt.xlim(xmin, xmax)`.

---

## Quick review (30 seconds)
- Gradient descent minimizes loss by stepping along the negative gradient.
- SGD noise + momentum + restarts help avoid stalls.
- Cross-entropy is large when the true class probability is low.
- Tokenization splits text into tokens; positional encoding injects order.
- Decoder self-attention is masked; encoder self-attention is typically unmasked.
- Python: `d[k]=v`, `set`, `sorted()` vs `.sort()`, `len()`.
- Matplotlib: scatter shows relationships; reduce overlap via `alpha` + jitter; use `subplot` for multiple axes.
