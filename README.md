🌀 Conditional 2D Diffusion Model (DDPM)
---

A PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) for generating complex 2D data manifolds such as Moons and Circles with high fidelity.  
The model supports Conditional Generation using Classifier-Free Guidance (CFG), enabling controlled sampling without relying on an external classifier.


---

🚀 Key Features
---

Architecture:
Custom Residual MLP with 512 hidden units, SiLU activations, and Layer Normalization.

Conditioning:
Classifier-Free Guidance implemented via vector arithmetic to steer sampling toward specific classes (e.g., Top Moon vs Bottom Moon).

Embeddings:
Learnable sinusoidal Time Embeddings and learned Class Embeddings.

Optimization:
Uses the Reparameterization Trick for O(1) forward noising and fully vectorized sampling loops.

Performance:
Achieved 99.8% class adherence on synthetic benchmarks.

---

🛠️ Installation
---

Clone this repository.
Look into SETUP.md for more insights.

---

💻 Usage
---

1️⃣ Training the Model
---

Train the model on Moons or Circles data:
```bash
dataset = MoonsDataset(n_samples=10000)  
model = ConditionalDDPM(n_dim=2, n_steps=200)  
trainConditional(model, dataset, epochs=1000, batch_size=128)
```
---

2️⃣ Inference (Sampling with CFG)
---

Generate samples for a specific class using Classifier-Free Guidance:
```bash
samples, labels = sampleCFG(
    model,
    n_samples=1000,
    guidance_scale=1.0,
    class_label=0
)
```
---

🧠 Model Architecture
---

The model backbone is a Residual MLP designed for non-spatial point cloud data.

Input:
```bash
[Noisy Data (2) + Time Embedding (256) + Class Embedding (256)]
```
```bash
Hidden Dimension:
512 neurons
```
```bash
Blocks:
2 Residual Blocks
Each block: Linear → LayerNorm → SiLU → Linear
Skip connections used for stability
```
Output:
```bash
Predicted noise epsilon_theta with dimension 2
```
---

📊 Results & Ablation Studies
---

The model was evaluated across different guidance scales (s), revealing a trade-off between sample diversity and class adherence.
```bash
Guidance Scale | Accuracy | Observation
0.0 (Uncond)   | ~50.0%   | Random generation (no control)
1.0 (Optimal)  | 99.8%    | High-fidelity, clean shapes
> 3.0          | ~100%    | Mode collapse (artifacts, line thinning)
```
---

📐 The Math (Simplified)
---

Classifier-Free Guidance modifies the predicted noise during inference:
```bash
epsilon_final = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

Where:
```bash
epsilon_uncond = unconditional prediction (general structure)
epsilon_cond   = conditional prediction (target class structure)
s              = guidance scale controlling strength of conditioning
```
---

📝 Notes
---

This project focuses on:
- Understanding diffusion dynamics in low-dimensional manifolds
- Controlled generation without classifiers
- Stability and efficiency of residual MLP-based diffusion models
