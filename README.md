This repository contains data and experiments associated to the paper: "Interpolation and Function Approximation Using Neural Networks and Barycentric Coordinates". Submited to the International Conference on Machine Learning (ICML 2025).

## Usage

To run the Jupyter notebooks correctly, you need to create a Python virtual environment with version 3.10:

```bash
python3 -m venv entorno python=3.10
```

Next, activate the virtual environment (we use the source command because we are working in WSL):

```bash
source entorno/bin/activate
```

Finally, install the required libraries:

```bash
pip install numpy matplotlib torch gudhi scipy ripser tensorflow
```

If Jupyter Notebook cannot find a kernel, you may need to update IPython. Run the following command to resolve this issue:

```bash
pip install --upgrade ipython
```

## Repository structure

- `utilsBaricentricNeuralNetwork.py`: Contains the implementation of the Baricentric Neural Network, as proposed in the paper, available in both PyTorch and TensorFlow frameworks.

- `utilsTopology.py`: Provides the necessary tools for topology and topological data analysis used in the experiments.

- `RepresentCPLF_BNN_Pytorch.ipynb`: Includes experiments on representing certain CPLFs (Continuous Piecewise Linear Functions) using the proposed Baricentric Neural Network in the PyTorch framework. An equivalent implementation in TensorFlow is available in `RepresentCPLF_BNN_Tensorflow.ipynb`.

- `AproxContinuousFunction_BNN_Pytorch.ipynb`: Contains experiments on approximating continuous functions using the proposed Baricentric Neural Network in the PyTorch framework.

- `OptimizePoints.ipynb`: Contains experiments aimed at optimizing the points used to approximate continuous functions with the Baricentric Neural Network.

## Citation and Reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es