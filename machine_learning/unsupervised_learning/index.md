---
layout: post
title: Unsupervised Learning
---


## Curse of dimensionality

Most datasets in real-case scenarios are *high-dimensional* with hundreds of features. For instance, the k-nearest neighbours algorithm would need exponentially more data points to maintaing a measure of "closeness" between them. The computation time also increases substantially in high dimensions, and the k-NN algorithm may become infeasible.

The k-NN classifier makes the assumption that similar points share similar labels. Unfortunately, in high dimensional spaces, points that are drawn from a probability distribution, tend to never be close together. We can illustrate this on a simple example. We will draw points uniformly at random within the unit cube and we will investigate how much space the $$k$$ nearest neighbors of a test point inside this cube will take up.

Formally, consider the unit cube $$[0,1]^d$$. All training data is sampled uniformly within this cube, i.e. $$\forall i, x_i \in [0,1]^d$$, and we are considering the $$k=10$$ nearest neighbors of such a test point.

Let $$l$$ be the edge length of the smallest hyper-cube that contains all $$k$$-nearest neighbor of a test point. Then $$l \approx (\frac{k}{n})^{\frac{1}{d}}$$.

If $$d=100$$, l=0.955$$; if $$d=1,000, l=0.995$$. In this case, the k-NN are not particularly closer (and therefore more similar) than any other data points in the training set. Why would the test point share the label with those k-nearest neighbors, if they are not actually similar to it?

One might think that one rescue could be to increase the number of training samples, $$n$$, until the nearest neighbors are truly close to the test point. However, in this case $$n=\frac{k}/{l^d}$$ would grow exponentially. This illustrates the principle of *curse of dimentionality*.

## Dimensionality reduction

We want to find a low-dimensional representation that captures the statistical properties of high-dimensional data.

- Data compression and visualisation
- Preprocessing before supervised learning (to improve performance of the model; regularization to reduce overfitting)
- Simplify the description of massive datasets by removing uninformative dimensions
- Reduce storage and computational costs

Low dimensional representations can then be used as feature vectors in machine learning algorithms.

Some dimensionality reduction methods include:

- Principal Components Analysis (PCA)
- Extensions of PCA, including Kernel PCA
- Nonnegative matrix factorisation (NMF)
- Linear discriminant analysis (LDA)
- Generalised discriminant analysis (GDA)
- Manifold learning approaches, including Local Linear Embedding

## Principal Component Analysis

Principal component analysis is a multivariate technique which allows to analye the statistical structure of high dimensional dependent observations by representing data using orthogonal variables called principal components.

Let $$(X_i)_{1\leq i\leq n}$$ be iid random variables in $$\mathbb{R}^d$$ and consider the matrix $$X\in\mathbb{R}^{n\times d}$$ such that the $$i$$-th row of $$X$$ is the observation $$X_i^T$$.

We assume that data are preprocessed so that the columns of $$X$$ are centered. Let $$\Sigma_n$$ be the empirical covariance matrix: $$\Sigma_n = \frac{1}{n}\Sigma_{i=1}^nX_iX_i^T$$

We can reduce the dimensionality of the observations $$(X_i)$$ using a compression matrix $$W \in \mathbb{R}^{p\times d}$$ with $$p\leq d$$ so that for each $$1\leq i \leq n, WX_i$$ is a low dimensional representation of $$X_i$$.

The original observation may then be partially recovered using another matrix $$U \in \mathbb{R}^{d\times p}$$.

PCA computes $$U$$ and $$W$$ using the least squares approach:

$$\left(U_{\star}, W_{\star}\right) \underset{(U, W) \in \mathbb{R}^{d \times p} \times \mathbb{R}^{p \times d}}{\mathop{\text{argmin}}} \Sigma_{i=1}^n\left\|X_i-U W X_i\right\|^2$$

Let $$(U_*, W_*) \in \mathbb{R}^{d \times p} \times \mathbb{R}^{p \times d}$$ be a solution. Then, the columns of $$U_*$$ are orthonormal and $$W_* = U_*^T$$.

For all $$U \in \mathbb{R}^{d \times p}$$ such that $$U^TU = I_p$$, we have:

$$\Sigma_{i=1}^n ||X_i - UU^TX_i||^2 = \Sigma_{i=1}^n ||X_i||^2 - \mathop{\text{trace}}(U^TXX^TU)$$.

Therefore, solving the PCA problem boils down to computing $$U_{\star} \in \underset{U \in \mathbb{R}^{d \times p}, U^T U=I_p}{\mathop{\text{argmax}}}{\mathop{\text{trace}}\left(U^T \Sigma_n U\right)}$$.

Let $$v_1, ..., v_d$$ be orthonormal eigenvectors associated with the eigenvalues $$\lambda_i \geq ... \geq \lambda_d$$ of $$\Sigma_n$$. Then a solution to the PCA problem is given by the matrix $$U_*$$ with columns $$v_1, ..., v_p$$ and $$W_* = U_*^T$$

- Compute the matrix $$\Sigma_n$$, obtain its eigenvectors sorted by eigenvalues
- Build the matrices $$U_*$$ and $$W_* = U_*^T$$
- Compressed data in $$\mathbb{R}^p$$ are given by $$W_*X_i$$ for all $$1\leq i \leq n$$.
