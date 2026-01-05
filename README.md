# NeuralNetwork_Scratch

Implementing the Neural Network from scratch and then implementing the same function using PyTorch

---

Table of contents
- [Project overview](#project-overview)
- [Goals](#goals)
- [What you'll find in this repository](#what-youll-find-in-this-repository)
- [Mathematical / conceptual summary](#mathematical--conceptual-summary)
- [Requirements](#requirements)
- [Quick start — run the notebooks](#quick-start--run-the-notebooks)
- [How the notebooks are organized](#how-the-notebooks-are-organized)
- [Key learning points](#key-learning-points)
- [Results and visualizations](#results-and-visualizations)
- [Extending the project](#extending-the-project)
- [Contributing](#contributing)
- [References and further reading](#references-and-further-reading)
- [License](#license)
- [Author / Contact](#author--contact)

## Project overview
This repository contains an educational implementation of a basic feedforward neural network implemented from scratch (using NumPy) and a parallel implementation using PyTorch. The goal is to show — step by step — how forward propagation, loss calculation, and backpropagation work, and then compare that low-level implementation to the higher-level PyTorch workflow.

This is a hands-on, notebook-first repository intended for learners who want to understand how neural networks work under the hood before using deep learning frameworks.

## Goals
- Implement a small neural network (MLP) from first principles (vectorized NumPy).
- Implement the same model using PyTorch and compare code clarity and performance.
- Visualize training dynamics (loss/accuracy curves), weight evolution, and decision boundaries for simple datasets.
- Serve as a teaching resource outlining both the math and the code.

## What you'll find in this repository
- Jupyter notebooks (main content) that:
  - Build a neural network from scratch: forward pass, backward pass, parameter updates.
  - Re-implement the model using PyTorch's `nn` API and `autograd`.
  - Train and evaluate on simple datasets (e.g., synthetic classification datasets, MNIST subset if included).
- Visualizations of training curves and decision boundaries.
- Explanations and inline commentary in the notebooks describing the math and rationale for implementation choices.

> Note: This repository is notebook-focused (Jupyter Notebook). Open the notebooks to read explanations and run the cells interactively.

## Mathematical / conceptual summary
The notebooks explain the following concepts and how they map to code:
- Feedforward neural networks (dense layers)
- Activation functions (Sigmoid, ReLU, Softmax)
- Loss functions (Mean Squared Error, Cross-Entropy)
- Backpropagation: computing gradients of loss w.r.t. weights and biases
- Gradient-based optimization (SGD, mini-batch updates)
- Weight initialization, regularization basics, and learning rate effects
- The difference between manual gradient computation and automatic differentiation (PyTorch)

## Requirements
A suggested minimal environment:
- Python 3.8+
- Jupyter / JupyterLab
- numpy
- matplotlib
- scikit-learn (for synthetic datasets and utilities)
- torch (if you want to run the PyTorch notebook)
- pandas (optional, for data handling)
- tqdm (optional, for progress bars)

Example (pip):
pip install -r requirements.txt

If this repo doesn't include `requirements.txt`, you can install the basics with:
pip install numpy matplotlib scikit-learn jupyterlab torch pandas tqdm

Or create a conda environment:
conda create -n nn-scratch python=3.9
conda activate nn-scratch
pip install numpy matplotlib scikit-learn jupyterlab torch pandas tqdm

## Quick start — run the notebooks
1. Clone the repository:
   git clone https://github.com/UditKandpal/NeuralNetwork_Scratch.git
   cd NeuralNetwork_Scratch

2. Create and activate a virtual environment (recommended).

3. Install dependencies (see Requirements section).

4. Start Jupyter:
   jupyter notebook
   or
   jupyter lab

5. Open the notebooks in the repository root (filenames will typically include "scratch" or "pytorch" in their names). Run cells sequentially and read the explanations.

Alternative: Convert notebooks to static HTML if you prefer:
jupyter nbconvert --to html path/to/notebook.ipynb

Notes:
- If a notebook expects datasets, either they will be included in the repo, or synthetic data will be generated inside the notebook using scikit-learn utilities such as `make_moons`, `make_circles`, or `make_classification`.
- Running the PyTorch notebook requires a CPU or GPU build of PyTorch. If you have CUDA, install the matching torch version.

## How the notebooks are organized
A typical structure for the notebooks:
- 00_introduction.ipynb — Overview and mathematical background
- 01_nn_from_scratch.ipynb — NumPy implementation: layer classes, forward/backward, training loop
- 02_nn_pytorch.ipynb — Equivalent model implemented using PyTorch: Dataset, DataLoader, model class, optimizer, training loop
- 03_experiments_and_visualizations.ipynb — Visual comparisons: loss/accuracy curves, decision boundaries, hyperparameter effects

(Actual file names may vary — open the repository to find the exact notebook names.)

## Key learning points
- Translating mathematical formulas (gradients) into efficient vectorized code
- Building a training loop with mini-batches, shuffling, and evaluation
- Understanding numerical stability issues (e.g., softmax + cross-entropy, avoiding overflow)
- Seeing how PyTorch's autograd drastically reduces boilerplate for gradient calculation
- Practical debugging tips: gradient checking, printing norms, small-scale experiments

## Results and visualizations
The notebooks include visual output such as:
- Training/validation loss and accuracy curves
- Decision boundaries for 2D toy datasets
- Weight histograms or weight-norm evolution plots
- Confusion matrices for classification tasks (if applicable)

Interpret these plots to understand underfitting/overfitting, learning rate effects, and model capacity.

## Extending the project
Ideas you can try:
- Add momentum, RMSProp, or Adam optimizers to the NumPy implementation.
- Extend to convolutional layers (from scratch implementation for conv layers is an excellent learning exercise).
- Implement regularization (L2 weight decay, dropout).
- Experiment with different activation functions (tanh, LeakyReLU).
- Train on a small real-world dataset (e.g., Fashion-MNIST).
- Add gradient checking (finite differences) to verify your backprop implementation.

## Contributing
Contributions are welcome! Ways to contribute:
- Fix typos or improve explanations in notebooks
- Add more experiments, visualizations, or comparison metrics
- Add a `requirements.txt` or environment.yml for reproducibility
- Add a CI workflow that runs notebook execution tests

If you open a PR, please:
- Keep notebook outputs cleared (or commit both .ipynb and generated HTML)
- Describe the change and include a short rationale

## References and further reading
- "Neural Networks and Deep Learning" — Michael Nielsen (interactive online book)  
- Goodfellow, Bengio, Courville — Deep Learning (textbook)  
- PyTorch official tutorials: https://pytorch.org/tutorials/  
- Stanford CS231n lectures (convolutional networks, backpropagation intuition)

## License
This repository is provided for educational purposes. If you'd like to use a permissive license, consider the MIT License:

MIT License — see LICENSE file (if included).  
(If you want a different license, update this section accordingly.)

## Author / Contact
Udit Kandpal — (github: [UditKandpal](https://github.com/UditKandpal))  

If you have questions or suggestions, please open an issue or submit a pull request. Thank you for checking out the project!

---

If you'd like, I can:
- generate a `requirements.txt` with pinned versions,
- add a small example script to run training from the command line,
- or produce a condensed `CONTRIBUTING.md` or `LICENSE` file (e.g., MIT).
Tell me which one you want next and I'll prepare it.
