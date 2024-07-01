---
layout: post
title: Bayesian Learning
---

The learning approaches we have discussed so far are based on the principle of maximum likelihood estimation. While being extremely general, there are limitations of this approach as illustrated in the two examples below.

## Example 1

Let's suppose we are interested in modeling the outcome of a biased coin, $$X \in \{heads, tails\}$$. We toss the coin 10 times, observing 6 heads. If $$\theta$$ denotes the probability of observing heads, the maximum likelihood estimate (MLE) is given by,

$$ \theta_{MLE} = \frac{n_\text{heads}}{n_\text{heads} + n_\text{tails}} = 0.6 $$

Now, suppose we continue tossing the coin such that after 100 total trials (including the 10 initial trials), we observe 60 heads. Again, we can compute the MLE as,

$$ \theta_{MLE} = \frac{n_\text{heads}}{n_\text{heads} + n_\text{tails}} = 0.6 $$

In both the above situations, the maximum likelihood estimate does not change as we observe more data. This seems counterintuitive - our _confidence_ in predicting heads with probability 0.6 should be higher in the second setting where we have seen many more trials of the coin! The key problem is that we represent our belief about the probability of heads $$\theta$$ as a single number $$\theta_{MLE}$$, so there is no way to represent whether we are more or less sure about $$\theta$$. 

## Example 2

Consider a language model for sentences based on the bag-of-words assumption. A bag of words model has a generative process where a sentence is formed from a sample of words which are metaphorically `pulled out of a bag', i.e. sampled independently. In such a model, the probability of a sentence can be factored as the probability of the words appearing in the sentence, i.e. for a sentence $$S$$ consisting of words $$w_1, \ldots, w_n$$, we have 

$$ p(S) = \prod_{i=1}^n p(w_n). $$

For simplicity, assume that our language corpus consists of a single sentence, "Probabilistic graphical models are fun. They are also powerful." We can estimate the probability of each of the individual words based on the counts. Our corpus contains 10 words with each word appearing once, and hence, each word in the corpus is assigned a probability of 0.1. Now, while testing the generalization of our model to the English language, we observe another sentence, "Probabilistic graphical models are hard." The probability of the sentence under our model is
$$0.1 \times 0.1 \times 0.1 \times 0.1 \times 0 = 0$$. We did not observe one of the words ("hard") during training which made our language model infer the sentence as impossible, even though it is a perfectly plausible sentence.

Out-of-vocabulary words are a common phenomena even for language models trained on large corpus. One of the simplest ways to handle these words is to assign a prior probability of observing an out-of-vocabulary word such that the model will assign a low, but non-zero probability to test sentences containing such words. As an aside, in modern systems, [tokenization](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html) is commonly used, where a set of fundamental tokens can be combined to form any word. Hence the word "Hello" as a single token and the word "Bayesian" is encoded as "Bay" + "esian" under the common Byte Pair Encoding. This can be viewed as putting a prior over all words, where longer words are less likely.

## Setup

In contrast to maximum likelihood learning, Bayesian learning explicitly models uncertainty over both the observed variables $$X$$ and the parameters $$\theta$$. In other words, the parameters $$\theta$$ are random variables as well.

A _prior_ distribution over the parameters, $$p(\theta)$$ encodes our initial beliefs. These beliefs are subjective. For example, we can choose the prior over $$\theta$$ for a biased coin to be uniform between 0 and 1. If however we expect the coin to be fair, the prior distribution can be peaked around $$\theta = 0.5$$. We will discuss commonly used priors later in this chapter.

If we observed the dataset $$\mathcal{D} = \lbrace x_1, \cdots, x_N \rbrace$$ (in the coin toss example, each $$X_i$$ is the outcome of one toss of the coin) we can update our beliefs using Bayes' rule,

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})} \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

$$
posterior \propto likelihood \times prior
$$

Hence, Bayesian learning provides a principled mechanism for incorporating prior knowledge into our model. Bayesian learning is useful in many situations such as when want to provide uncertainty estimates about the model parameters (Example 1) or when the data available for learning a model is limited (Example 2).
