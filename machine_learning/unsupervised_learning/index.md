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

## Expectation-maximization (EM) algorithm

We consider statistical models where we do not have a complete data set of observations from $$Z$$.

We will assume that the data $$Z$$ consists of *observed data* $$Y = (Y_1, ..., Y_k)$$ and *latent* (or *missing*) *data* $$X = (X_1, ..., X_k)$$. We assume that there exists $$\theta \in \mathbb{R}^m$$ and a probability density $$(x,y) \mapsto p_{\theta}(x,y)$$.

With this notation, the log-likelihood function for the observed data $$Y$$ is

$$\log p_{\theta}(Y) = \log \int p_{\theta}(x,Y)dx$$

As we don’t measure $$X$$, we cannot maximize $$\theta \mapsto \log p_{\theta}(X,Y)$$. We only have access to $$Y$$ and thus $$\theta \mapsto \log p_{\theta}(Y)$$.

The main difficulty is we cannot compute $$p_{\theta}(y) = \int p_{\theta}(x,y)dx$$ so we can't maximize it directly. The EM algorithm instead computes an auxiliary quantity, generating a sequence of estimators $$(\theta^{(p)})_{p \geq 0}$$ which optimize {% include sidenote.html id="note-em" note="The EM algorithm may converge to a *local* but not necessarily *global* maximum." %} the log-likelihood as follows:

1. $$\theta^{(0)}$$: randomly initialized.
2. $$\forall k \geq 0$$: Compute
   $$Q(\theta; \theta^{(k)}) = E_{\theta^{(k)}}[\log p_{\theta}(X,Y) \mid Y] = \int \log p_{\theta}(x,Y) p_{\theta^{(k)}}(x \mid Y) \, dx$$ *(E-step)*.
3. Define $$\theta^{(k+1)} \in \operatorname{Argmax}_{\theta \in \mathbb{R}^m} Q(\theta; \theta^{(k)})$$ *(M-step)*.

The procedure is iterated until the algorithm converges. The EM algorithm always increases the likelihood, that is $$\log p_{\theta^{(k+1)}}(Y) \geq \log p_{\theta^{(k)}}(Y)$$. Indeed, we have through Bayes' rule the following result:

$$\log p_{\theta}(Y) = Q(\theta, \theta^{(k)}) - E_{\theta^{(k)}}[\log p_{\theta}(X|Y)|Y]$$

Thus,

$$\log p_{\theta}(Y) - \log p_{\theta^{(k)}}(Y) = Q(\theta, \theta^{(k)}) - Q(\theta^{(k)}, \theta^{(k)}) + E_{\theta^{(k)}}[\log p_{\theta^{(k)}}(X|Y)|Y] - E_{\theta^{(k)}}[\log p_{\theta}(X|Y)|Y]$$

But

$$
E_{\theta^{(k)}}[\log p_{\theta^{(k)}}(X \mid Y) \mid Y] - E_{\theta^{(k)}}[\log p_{\theta}(X \mid Y) \mid Y] = E_{\theta^{(k)}}\left[\log \frac{p_{\theta^{(k)}}(X \mid Y)}{p_{\theta}(X \mid Y)} \mid Y \right] := KL\left(p_{\theta^{(k)}}(\cdot \mid Y) \| p_{\theta}(\cdot \mid Y)\right)
$$


The latter quantity is known as the *Kullback-Leibler divergence*, and is positive by definition. Indeed, since $$-\log$$ is convex, we have:
$$KL\left(p_{\theta^{(k)}}(\cdot \mid Y) \| p_{\theta}(\cdot \mid Y)\right) = E_{\theta^{(k)}}\left[-\log \frac{p_{\theta}(X|Y)}{p_{\theta^{(k)}}(X|Y)} \mid Y \right] \geq -\log E_{\theta^{(k)}}\left[\frac{p_{\theta}(X|Y)}{p_{\theta^{(k)}}(X|Y)} \mid Y \right] = 0$$

In addition, since $$\theta^{(k+1)} \in \operatorname{Argmax}_{\theta \in \mathbb{R}^m} Q(\theta; \theta^{(k)})$$, we have: $$Q(\theta^{(k+1)}, \theta^{(k)}) - Q(\theta^{(k)}, \theta^{(k)}) \geq 0$$,
which concludes the proof.

Indeed, the previous response provides a general overview of the Expectation-Maximization (EM) algorithm applied to Gaussian Mixture Models using your specified notation, but it did not include the detailed derivations for updating the parameters $$\mu$$ and $$\Sigma$$. Let's include these derivations now, making sure we work within the structure and format you're using:

## Detailed Derivations for M-Step in Gaussian Mixture Models

The Expectation (E) step calculates the posterior probabilities (responsibilities) $$\gamma(z_{nk})$$, which represent the probability that the $$n$$-th observation is generated by the $$k$$-th Gaussian component. The Maximization (M) step then updates the parameters based on these responsibilities.

### Updating $$\mu_k$$

To derive the update for the means ($$\mu_k$$) of each Gaussian component, we consider the Q function defined in the E-step:

$$
Q(\theta; \theta^{(k)}) = \mathbb{E}_{\theta^{(k)}}[\log p_{\theta}(X, Y) | Y]
$$

Breaking it down for the Gaussian Mixture Model, and focusing on the $$\mu_k$$ update, we derive:

$$
Q(\theta; \theta^{(k)}) = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \log \left( \pi_k \mathcal{N}(y_n | \mu_k, \Sigma_k) \right)
$$

For Gaussian distributions:

$$
\mathcal{N}(y_n | \mu_k, \Sigma_k) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_k|}} \exp \left(-\frac{1}{2} (y_n - \mu_k)^T \Sigma_k^{-1} (y_n - \mu_k)\right)
$$

To maximize $$Q$$ with respect to $$\mu_k$$, we take the derivative of $$Q$$ with respect to $$\mu_k$$ and set it to zero:

$$
\frac{\partial Q}{\partial \mu_k} = \sum_{n=1}^N \gamma(z_{nk}) \Sigma_k^{-1} (y_n - \mu_k) = 0
$$

This yields:

$$
\sum_{n=1}^N \gamma(z_{nk}) y_n = \sum_{n=1}^N \gamma(z_{nk}) \mu_k
$$

Therefore, the updated mean $$\mu_k$$ is:

$$
\mu_k = \frac{\sum_{n=1}^N \gamma(z_{nk}) y_n}{\sum_{n=1}^N \gamma(z_{nk})}
$$

### Updating $$\Sigma_k$$

Similarly, to update the covariance matrices $$\Sigma_k$$, we maximize $$Q$$ by taking the derivative with respect to $$\Sigma_k$$ and setting it to zero. The derivation considers the second term in the exponential of the Gaussian density:

$$
\frac{\partial Q}{\partial \Sigma_k} = \sum_{n=1}^N \gamma(z_{nk}) \left[ -\frac{1}{2} \Sigma_k^{-1} + \frac{1}{2} \Sigma_k^{-1} (y_n - \mu_k) (y_n - \mu_k)^T \Sigma_k^{-1} \right] = 0
$$

Solving this gives:

$$
\Sigma_k = \frac{\sum_{n=1}^N \gamma(z_{nk}) (y_n - \mu_k) (y_n - \mu_k)^T}{\sum_{n=1}^N \gamma(z_{nk})}
$$

### Conclusion

The derived formulas for $$\mu_k$$ and $$\Sigma_k$$ are used in the M-step to update the parameters of the Gaussian components based on the current estimates of the responsibilities $$\gamma(z_{nk})$$ computed in the E-step. This process iterates until convergence, ensuring each step either improves or retains the likelihood of the observed data under the model. You can check my streamlit [application](https://emalgorithm-m9upjzdzpacfujan7cvnqr.streamlit.app/) for a visual illustration of the EM algorithm.
