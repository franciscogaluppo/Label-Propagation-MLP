# Label  Propagation  with  Learnable  Weights using MLP
Code repository for the research of a nascent model of semi-supervised machine learn algorithm. In this model we use a Multilayer Percepetron (MLP) to train the Label Propagation algorithm for vertex classification with edge features. The main limitation of the model is its lack of scalability as no modern machine learning framework handles its need of sparsity.

All of the researchers are from the Computer Science Department at Universidade Federal de Minas Gerais (UFMG), Brazil:

Francisco Galuppo Azevedo - franciscogaluppo@dcc.ufmg.br

Fabricio Murai - murai@dcc.ufmg.br

## Our paper

*No paper yet.*

## Dependencies
To run our code, you will need the following libraries:

```bash
pip install !!!!!!
```

## How to run our code
The easiest way to use our algorithms is to test them with a random graph:

```bash
python3 main.py
```

## Label Propagation
[Label Propagation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf) is a learning algorithm that assigns labels to unlabeled vertices. At the start of the algorithm, a small subset of the vertices have labels. These labels are then propagated to the unlabeled vertices. Intuitively, we want vertices that are close to have similar labels.

![](videos/README.gif?raw=true)

The key point of the algorithm is to propagate a node's label to all nodes according to their proximity, while fixing the labels on the labeled data.  Labeled data act like sources that dissipate labels through unlabeled data.

## MLP classifier
A multilayer perceptron (MLP) is a class of feedforward artificial neural network, which utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation differentiate MLP from a linear perceptron, being thus able to distinguish data that is not linearly separable.

## The algorithm
*To be written...*

## Different implementations
All of our implementations were in *Python*, a powerful language to implement the above algorithm, given that the most used machine learning frameworks are developed for it. During our work, we used three different frameworks, each one with its pros and cons: *MXNet*, *TensorFlow* and *PyTorch*.\\

*MXNet* offered a simple and intuitive way to implement our model, but lacked sparsity support. As for *TensorFlow* and *PyTorch*, there was no support for backpropagation to handle the sparse matrices. With the current frameworks, there is no scalable solution for the algorithm.

## Results
*To be written...*
