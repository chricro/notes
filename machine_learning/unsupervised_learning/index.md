---
layout: post
title: Unsupervised Learning
---


## Curse of dimensionality

Most datasets in real-case scenarios are *high-dimensional* with hundreds of features. For instance, the k-nearest neighbours algorithm would need exponentially more data points to maintaing a measure of "closeness" between them. The computation time also increases substantially in high dimensions, and the k-NN algorithm may become infeasible.

The k-NN classifier makes the assumption that similar points share similar labels. Unfortunately, in high dimensional spaces, points that are drawn from a probability distribution, tend to never be close together. We can illustrate this on a simple example. We will draw points uniformly at random within the unit cube and we will investigate how much space the $$k$$ nearest neighbors of a test point inside this cube will take up.

Formally, consider the unit cube $$[0,1]^d$$. All training data is sampled uniformly within this cube, i.e. $$\forall i, x_i \in [0,1]^d$$, and we are considering the $$k=10$$ nearest neighbors of such a test point.

Let $$l$$ be the edge length of the smallest hyper-cube that contains all $$k$$-nearest neighbor of a test point. Then $$l \approx (\frac{k}{n})^{\frac{1}{d}}$$.

If $$d=100, l=0.955$$; if $$d=1,000, l=0.995$$. In this case, the k-NN are not particularly closer (and therefore more similar) than any other data points in the training set. Why would the test point share the label with those k-nearest neighbors, if they are not actually similar to it?

One might think that one rescue could be to increase the number of training samples, $$n$$, until the nearest neighbors are truly close to the test point. However, in this case $$n=\frac{k}{l^d}$$ would grow exponentially. This illustrates the principle of *curse of dimentionality*.

## Dimensionality reduction

We want to find a low-dimensional representation that captures the statistical properties of high-dimensional data.

The main applications are among the following:

- Compress and visualize data
- Preprocess data before supervised learning to improve model performance and reduce overfitting
- Simplify the description of massive datasets by removing uninformative dimensions
- Reduce storage and computational costs

Some dimensionality reduction methods include *Principal Components Analysis* (PCA), *Kernel PCA*, *Nonnegative Matrix Factorisation* (NMF), *Linear Discriminant Analysis* (LDA), *Generalised Discriminant Analysis* (GDA), *Nonlinear independent component analysis* (ICA), *Uniform manifold approximation* (UMA), *Locally Linear Embedding* (LLE), and also *Random Forests*.

## Principal Component Analysis (PCA)

Principal component analysis is a multivariate technique which allows to analye the statistical structure of high dimensional dependent observations by representing data using orthogonal variables called principal components.

There are several equivalent ways of deriving the principal components mathematically. The simplest one is by finding the projections which maximize the variance. The first principal component is the direction in space along which projections have the largest variance. The second principal component is the direction which
maximizes variance among all directions orthogonal to the first. The $$k$$th component is the variance-maximizing direction orthogonal to the previous $$k-1$$ components. There are $$p$$ principal components in all.
Rather than maximizing variance, it might sound more plausible to look for the projection with the smallest average (mean-squared) distance between the original vectors and their projections on to the principal components; this turns out to be equivalent to maximizing the variance.

Let $$(X_i)_{1\leq i\leq n}$$ be iid random variables in $$\mathbb{R}^d$$ and consider the matrix $$X\in\mathbb{R}^{n\times d}$$ such that the $$i$$-th row of $$X$$ is the observation $$X_i^T$$.

We assume that data are preprocessed so that the columns of $$X$$ are centered. Let $$\Sigma_n$$ be the empirical covariance matrix:

$$\Sigma_n = \frac{1}{n}\Sigma_{i=1}^nX_iX_i^T$$

We can reduce the dimensionality of the observations $$(X_i)$$ using a compression matrix $$W \in \mathbb{R}^{p\times d}$$ with $$p\leq d$$ so that for each $$1\leq i \leq n, WX_i$$ is a low dimensional representation of $$X_i$$.

The original observation may then be partially recovered using another matrix $$U \in \mathbb{R}^{d\times p}$$.

PCA computes $$U$$ and $$W$$ using the least squares approach:

$$\left(U_{\star}, W_{\star}\right) \underset{(U, W) \in \mathbb{R}^{d \times p} \times \mathbb{R}^{p \times d}}{\mathop{\text{argmin}}} \Sigma_{i=1}^n\left\|X_i-U W X_i\right\|^2$$

Let $$(U_*, W_*) \in \mathbb{R}^{d \times p} \times \mathbb{R}^{p \times d}$$ be a solution. Then, the columns of $$U_*$$ are orthonormal and $$W_* = U_*^T$$.

For all $$U \in \mathbb{R}^{d \times p}$$ such that $$U^TU = I_p$$, we have:

$$\sum_{i=1}^n \|X_i - UU^T X_i\|^2 = \sum_{i=1}^n \|X_i\|^2 - \operatorname{trace}(U^T X X^T U)$$.

Therefore, solving the PCA problem boils down to computing

$$U_{\star} \in \underset{U \in \mathbb{R}^{d \times p}, U^T U = I_p}{\operatorname{argmax}} \operatorname{trace}\left(U^T \Sigma_n U\right)$$

Let $$v_1, ..., v_d$$ be orthonormal eigenvectors associated with the eigenvalues $$\lambda_1 \geq ... \geq \lambda_d$$ of $$\Sigma_n$$. Then a solution to the PCA problem is given by the matrix $$U_*$$ with columns $$\lbrace v_1, ..., v_p \rbrace $$.
Here is the pseudo-code of the PCA algorithm:

1. Center $$X\in\mathbb{R}^{n\times d}$$
2.  Compute the covariance matrix $$\Sigma_n$$, obtain its eigenvectors $$v_i \in \mathbb{R}^{d}$$ sorted by eigenvalues in decreasing order
3.  Build the matrice $$U_* = (v_1, ..., v_p) \in \mathbb{R}^{d \times p}$$ by stacking the $$p$$ eigenvectors $$v_i$$ alongside one another.
4.  Compressed data are given by $$Z = XU_* \in \mathbb{R}^{n \times p}$$

PCA only allows dimensionality reduction based on principal components which are linear combinations of the variables. When the data has more complex structures which cannot be well represented in a linear subspace, standard PCA fails. Kernel PCA allows us to generalize standard PCA to nonlinear dimensionality reduction.

## Cluster analysis

* The *K-means algorithm* classifies $$n$$ points into $$k$$ clusters in a vector space, based on their distance to each other. It starts by randomly chosing the representant of each cluster-sometimes referred as *centroids*, and then iteratively assigns each of the **n** points to its closest centroid to obtain **k** clusters. The centroid of the latters will be updated as the barycenter of the cluster's points. The algorithm goes on until it converges -which is when centroids remain the same between two consecutive iterations.

* The *K-medoids* is almost the same as *K-means*, the only difference relies in the definition of the centroids. The representant of the clusters are not defined as the barycenter but as the point of the cluster that minimises the average distance with its class neighbours. Therefore, it corresponds to a datapoint which wouldn't be always the case with K-Means.

## Expectation-maximization algorithm

We consider statistical models where **Y** are *observed data* and **X** are *latent* (or *missing*) *data*. We assume that there exists **\theta \in \mathbb{R}^m** and a probability density **(x,y) \mapsto p_{\theta}(x,y)**.

As we don't measure **X**, we cannot maximize **\theta \mapsto \log p_{\theta}(X,Y)**. We only have access to **Y** and thus **\theta \mapsto \log p_{\theta}(Y). Suppose the completed data has a density **f**

The main difficulty is we cannot compute **p_{\theta}(y) = \int p_{\theta}(x,y)dx**. The EM algorithm instead computes an auxiliary quantity, generating a sequence of estimators **(\theta^{(p)})_{p\geq 0}** with:
- **\theta^{(0)}** randomly initialized
- **\forall k \geq 0**: compute **Q(\theta; \theta^{(k)}) = E_{Q^{(k)}}[\log p_{\theta}(X,Y)|Y] = \int \log p_{\theta}(x,Y)p_{\theta^{(k)}}(x|Y)dx** (E-step)
- define **\theta^{(k+1)} \in \operatorname{Argmax}_{\theta \in \mathbb{R}^m} Q(\theta; \theta^{(k)}**

The EM algorithm always increases the likelihood:, that is **\log p_{\theta^{(k+1)}}(y) \geq \log p_{\theta^{(k)}(y)**.
Indeed, we have:

**\log p_{\theta}(Y) = Q(\theta, \theta^{(k)}) - E_{\theta^{(k)}}[\log p_{\theta}(X|Y)|Y]**
Thus,
**\log p_{\theta}(y) \geq \log p_{\theta^{(k)}(y) = Q(\theta,\theta^{(k)})-(\theta^{(k)},\theta^{(k)})+E_{\theta^{(k)}}[\log p_{\theta}(X|Y)|Y]-E_{\theta^{(k)}}[\log p_{\theta^{(k)}}(X|Y)|Y]**
But
**E_{\theta^{(k)}}[\log p_{\theta}(X|Y)|Y]-E_{\theta^{(k)}}[\log p_{\theta^(k)}(X|Y)|Y] = E_{\theta^{(k)}}[\log \frac{p_{\theta}(X|Y)}{p_{\theta^{(k)}}|Y}]** is positive as a Kullback divergence.
In addition, by definition: **Q(\theta^{(k+1)}, \theta^{(k)}) - Q(\theta^{(k)}, \theta^{(k)}) \leq 0**, which concludes the proof.
