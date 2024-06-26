---
layout: post
title: Supervised Learning
---

## Classification:

There are two main approaches for doing classification: the *generative approach* and the *discriminative approach*.

1. The generative approach models the class-conditional densities $$p(x|y = k)$$ as well as the class priors $$p(y = k)$$, and uses these to compute posterior probabilities $$p(y = k|x)$$ with Bayes’ theorem. We can use this approach to generate typical data from the model by drawing samples from $$p(x|y = k)$$.
Examples: LDA, QDA, Naive Bayes, etc.

* The Bayes Classifier chooses the class with the greatest probability given the observation, i.e. $$\text{argmax}_k p(y = k|x) = \text{argmax}_k p(x|y = k)p(y = k)$$.
In general, we known neither the conditional densities $$p(x|y = k)$$ nor the class probabilities $$p(y = k)$$. The plug-in classifier uses estimate of these probabilities.

* The discriminant classifiers partitions $$X = \mathbb{R}^d$$ into regions with the same class predictions via separating hyperplanes. The conditional densities are modeled as multivariate normal. For all class $$k$$, conditionnally on $$\lbrace Y = k \rbrace$$,
$$X \sim N(\mu_k,\Sigma_k)$$.

The discriminant functions are given by: $$g_k(X) = log(P(X|Y=k) + log(P(Y=k))$$.
In a two-classes problem, the optimal classifier is $$f^*: x \mapsto 2 \mathbb{1}\lbrace g_1(x) - g_{-1}(x) \rbrace -1$$.

*Linear Discriminant Analysis* (LDA) assumes $$\Sigma_k = \Sigma$$ for all $$k$$; *Quadratic Discriminant Analysis* (QDA) assumes differents $$\Sigma_k$$ in each class (the decision boundaries are hence quadratic.

* The Naive Bayes classifier is another plug-in classifier with a simple generative model: it "naïvely" assumes all measured variables/features are conditionally independent given the class label:
$$p(x|y = k) = \prod_{i=1}^d p(x_i|y = k,\theta_{ik})$$. One advantage of this classifier is that it allows to easily mix and match different types of features and handle missing data. It is often used with categorical data, e.g. text document classification. The form of the class-conditional density depends of the type of each features.
Examples:
For real-valued features, the Gaussian distribution can be used $$p(x|y=k'θ)=\prod_{j=1}^d \Phi(x|\mu ,\sigma^2)$$.
For binary features, $$x_j \in  \lbrace 0, 1 \rbrace$$, the Bernoulli distribution can be used: $$p(x|y = k′\theta) = \prod_{j=1}^d Ber(x_j|\theta_{jk})$$. This is called the *Bernoulli Naive Bayes model*.
For count data, $$x_j \in \lbrace 0, 1, 2, ... \rbrace$$, the multinomial distribution can be used. This is called the *Multinomial Naive Bayes model*.

2. An alternative approach is to not model the class-conditional density $$p(x|y = k)$$ at all, and assume a functional form of a generalised linear model for the discriminant function directly. The discriminative approach aims to model the conditional probability $$p(y = k|x)$$ directly, for example using a linear model like the one we used for regression. We can consider this by forming a likelihood based on the discriminant function.
Examples: logistic regression, K-nearest neighbors, SVMs, perceptrons, etc. 

The objective is to predict the label $$Y\in \lbrace 0, 1 \rbrace$$ based on$$X \in \mathbb{R}^d$$. Logistic regression models the distribution of $$Y$$ given $$X$$.
$$P(Y=1|X) = \sigma(<w,X>+b)$$ where $w \in \mathbb{R}^d$$ is a vector of model weights and $$b \in \mathbb{R}$$ is the intercept, and $$\sigma$$ is the sigmoid function $$\sigma: z \mapsto \frac{1}{1+e^{-z}}$$

We define the log-odd ratio as: $$log(P(Y=1|X)) - log(P(Y=0|X)) = <w, X> +b$$. Thus, we have $$P(Y=1|X) \geq P(Y=0|X) \iif <w,X> +b$$, defining our classification rule (linear classification rule) which requires to estimate $$w$$ and $$b$$.
These latter parameters can be estimated by Maximum Likelihood estimation.

For a multi-label classification, we can extend the logistic regression. The objective is to predict the label $$Y \in \lbrace 1, ..., M \rbrace based on $$X \in \mathbb{R}^d$$. Softmax regression models the distribution of $$Y$$ given $$X$$.

For all $$1 \leq m \leq M, z_m = <w_m,X> + b_m$$ and $$P(Y=m|X)=\text{softmax}(z)_m$$ where $$z\in \mathbb{R}^M, w_m\in \mathbb{R}^d$$ is a vector of model weights and $$b_m \in \mathbb{R}$$ is an intercept and $$\text{softmax}(z)_m = \frac{exp(z_m)}{\Sigma_{j=1}^M exp(z_j)}$$. One neuron is a multi-class extension of the logistic regression model.

## Regression:


