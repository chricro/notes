---
layout: post
title: Contents
---
<span class="newthought">This website</span> is the collection of notes about diverse topics from statistics, machine learning, artificial intelligence but also finance {% include sidenote.html id="note-pgm" note="This is just an example of side notes" %}.

## Statistical learning

Statistical learning refers to a set of tools for modeling and understanding complex datasets. It is a recently developed area in statistics and blends with parallel developments in computer science and, in particular, machine learning. The field encompasses many methods such as the lasso and sparse regression, classification and regression trees, and boosting and support vector machines.
With the explosion of “Big Data” problems, statistical learning has become a very hot field in many scientific areas as well as marketing, finance, and other business disciplines.

Here is a combination of different elements that I learned from Machine Learning courses at Télécom SudParis and Imperial College London:

- [Supervised learning](machine_learning/supervised_learning/): Generative approache (LDA/GDA, Naive Bayes) vs discriminant approach (logistic regression)
- [Unsupervised learning](machine_learning/unsupervised_learning/): Principal Component Analysis (PCA), Kernel PCA

## Generative AI

Here are some self-taught notes about Generative AI I've written during my full-time role in a startup environment.

- [An overview of large language models](ai/llm/): Pretraining, instruction fine-tuning, alignment techniques. Frameworks to deploy LLMs efficiently, optimization of inference. How to improve reasoning. Vision transformers.
- [Useful resources](ai/resources/): Benchmarks and leaderboards, interesting blogs and github resources.

## Finance

As someone interested in applying my statistical and machine learning knowledge on concrete problems and real-world scenarios, I find finance to be a passioning field for research and exploration.

1. [Book Excerpts](finance/books/): Sentences from books that I find interesting and educational.


## Stanford CS228:

{% include marginnote.html id='mn-construction' note='This part of the website is **not written by me**!'%}

Preliminaries:

1. [Introduction](preliminaries/introduction/): What is probabilistic graphical modeling? Overview of the course.

2. [Review of probability theory](preliminaries/probabilityreview): Probability distributions. Conditional probability. Random variables (*under construction*).

3. [Real-world applications](preliminaries/applications): Image denoising. RNA structure prediction. Syntactic analysis of sentences. Optical character recognition. Language Modeling (*under construction*).

Representation:

1. [Bayesian networks](representation/directed/): Definitions. Representations via directed graphs. Independencies in directed models.

2. [Markov random fields](representation/undirected/): Undirected vs directed models. Independencies in undirected models. Conditional random fields.

Inference:

1. [Variable elimination](inference/ve/) The inference problem. Variable elimination. Complexity of inference.

2. [Belief propagation](inference/jt/): The junction tree algorithm. Exact inference in arbitrary graphs. Loopy Belief Propagation.

3. [MAP inference](inference/map/): Max-sum message passing. Graphcuts. Linear programming relaxations. Dual decomposition.

4. [Sampling-based inference](inference/sampling/): Monte-Carlo sampling. Forward Sampling. Rejection Sampling. Importance sampling. Markov Chain Monte-Carlo. Applications in inference.

5. [Variational inference](inference/variational/): Variational lower bounds. Mean Field. Marginal polytope and its relaxations.

Learning:

1. [Learning in directed models](learning/directed/): Maximum likelihood estimation. Learning theory basics. Maximum likelihood estimators for Bayesian networks.

2. [Learning in undirected models](learning/undirected/): Exponential families. Maximum likelihood estimation with gradient descent. Learning in CRFs

3. [Learning in latent variable models](learning/latent/): Latent variable models. Gaussian mixture models. Expectation maximization.

4. [Bayesian learning](learning/bayesian/): Bayesian paradigm. Conjugate priors. Examples (*under construction*).

5. [Structure learning](learning/structure/): Chow-Liu algorithm. Akaike information criterion. Bayesian information criterion. Bayesian structure learning (*under construction*).
