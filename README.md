# **WR-CMLP method: One-step Cost Classification**

This repository presents the code of WR-CMLP: Cost MultiLayer Perceptron or Cost Neural Network method. Also contains the code to reproduce experiments with one of the real-world datasets presented in the original paper. The rest of the datasets presented in the paper are publicly available on the repository: https://github.com/JavierMediaRB/EDCcounterfactuals

>The WR-CMLP method trains a Neural Network classifier to solve example-dependent cost classification when the classification costs are not available on production (for unseen samples). The novel WR-CMLP method proposes to train the Neural Network model with a cost-normalized and weighted loss function to obtain a model that learns to classify accordingly to minimize the classification costs without using them. The proposed loss function is the following:

<p align="center">
  <image src="images/cost_loss_function.png" alt="" style="width: 800px;" >
</p>

<p style="text-align: center;">
The training loss function for the WR-CMLP method
</p>

<p align="center">
  <image src="images/tral_loss_and_savings.png" alt="" style="width: 500;" >
</p>

<p style="text-align: center;">
As the proposed cost-normalized and weighted loss function is minimized during training, the cost savings are maximized.
</p>

A more detailed explanation of the WR CMLP method is provided in the [Paper Citation](#Citations).

# **Table of Contents**

* [1. Installation and Usage](#installation-and-usage)
* [2. WR-CMLP Method](#WR-CMLP-method)
* [3. Paper Citation](#Citation)

# **1. Installation and Usage**
The code can be downloaded directly from this repository. Also, to run the code you must create an environment for the WR-CMLP method. The configuration of the environment can be done with Anaconda or Miniconda.

- WR-CMLP method: The novel one-step costs classification method

The WR-CMLP method has a notebook to reproduce the load, training and prediction for the CS2 dataset. The notebook must be executed with the specified environment.

The notebook:
- WR-CMLP: example_tutorial_WR-CMLP_method.ipynb ; execute with [env_WR_CMLP_method](#Environment) environment

## 1.1. Environment for WR-CMLP method

- Create a new environment with conda:

  ```bash
  conda create -n env_WR_CMLP_method python=3.8.13
  ```
  
- Activate the environment:
  ```bash
  conda activate env_WR_CMLP_method
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_WR_CMLP.txt
  ```


## 1.2. Usage

You can just import the class WR_CMLP_model from WR_CMLP_model.py to use the WR CMLP method on your code. To facilitate the reproducibility of the proposed method, also a notebook is provided as an example of how to create, train, and predict with the WR CMLP method on a dataset.
The notebook is organised with the following sections:

- *Section 1. Libraries*
    > Install the dependences

- *Section 2. Load data*
    > Load the dataset to train and test the WR CMLP method

- *Section 3. CMLP Method*

    > Trains the WR CMLP method and shows an example of how to make predictions and decisions with it.


# **2. Paper Citation**
If you use the WR CMLP method in your research, please consider citing it.

BibTeX entry:

```
...
```

