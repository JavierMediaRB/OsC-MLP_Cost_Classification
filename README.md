# **OsC-MLP method: One-step Cost Classification**

This repository presents the code of OsC-MLP: Cost MultiLayer Perceptron or Cost Neural Network method. Also contains the code to reproduce experiments with one of the real-world datasets presented in the original paper. The rest of the datasets presented in the paper are publicly available on the repository: https://github.com/JavierMediaRB/EDCcounterfactuals

>The OsC-MLP method trains a Neural Network classifier to solve example-dependent cost classification when the classification costs are not available on production (for unseen samples). The novel OsC-MLP method proposes to train the Neural Network model with a cost-normalized and weighted loss function to obtain a model that learns to classify accordingly to minimize the classification costs without using them. The proposed loss function is the following:

<p align="center">
  <image src="images/cost_loss_function.png" alt="" style="width: 800px;" >
</p>

<p style="text-align: center;">
The training loss function for the OsC-MLP method
</p>

<p align="center">
  <image src="images/tral_loss_and_savings.png" alt="" style="width: 500;" >
</p>

<p style="text-align: center;">
As the proposed cost-normalized and weighted loss function is minimized during training, the cost savings are maximized.
</p>

A more detailed explanation of the OsC-MLP method is provided in the [Paper Citation](#Citations).

# **Table of Contents**

* [1. Installation and Usage](#installation-and-usage)
* [2. OsC-MLP Method](#OsC-MLP-method)
* [3. Paper Citation](#Citation)

# **1. Installation and Usage**
The code can be downloaded directly from this repository. Also, to run the code you must create an environment for the OsC-MLP method. The configuration of the environment can be done with Anaconda or Miniconda.

- OsC-MLP method: The novel one-step costs classification method

The OsC-MLP method has a notebook to reproduce the load, training and prediction for the CS2 dataset. The notebook must be executed with the specified environment.

The notebook:
- OsC-MLP: example_tutorial_OsC_MLP_method.ipynb ; execute with [env_OsC_MLP_method](#Environment) environment

## 1.1. Environment for OsC-MLP method

- Create a new environment with conda:

  ```bash
  conda create -n env_OsC_MLP_method python=3.8.13
  ```
  
- Activate the environment:
  ```bash
  conda activate env_OsC_MLP_method
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_OsC_MLP.txt
  ```


## 1.2. Usage

You can just import the class OsC_MLP_model from OsC_MLP_model.py to use the OsC MLP method on your code. To facilitate the reproducibility of the proposed method, also a notebook is provided as an example of how to create, train, and predict with the OsC MLP method on a dataset.
The notebook is organised with the following sections:

- *Section 1. Libraries*
    > Install the dependences

- *Section 2. Load data*
    > Load the dataset to train and test the OsC-MLP method

- *Section 3. OsC-MLP Method*

    > Trains the OsC-MLP method and shows an example of how to make predictions and decisions with it.


# **2. Paper Citation**
If you use the OsC-MLP method in your research, please consider citing it.

BibTeX entry:

```
@article{mediavilla2024one,
  title={One-step Bayesian example-dependent cost classification: The OsC-MLP method},
  author={Mediavilla-Rela{\~n}o, Javier and L{\'a}zaro, Marcelino},
  journal={Neural Networks},
  pages={106168},
  year={2024},
  publisher={Elsevier}
}
```

