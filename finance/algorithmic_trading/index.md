---
layout: post
title: Algorithmic trading
---

Algorithmic Trading is an automated process taking input signals and producing trades as outputs.

A trading algorithm takes input signals, often called *alphas*, and outputs trades.

The trading infrastructures are complex because we first need to make sure the algorithm works as expected; it should scale and respect latency constraints and feedback loops; and need to be backtested (simulation) and live trading tested (production) to ensure the model is still valid.

- Quantitative developers build production and backtests tools.
- Quantitative researchers use the tools to build strategies and signals on massive datasets.
- Quantitative traders measure and monitor algorithms and signals daily.

## Price impact

Price impact is an expression used to describe the correlation between an incoming order and the change in the price of the asset involved caused by the trade. Buy trades push the price of a given asset higher by exhausting the cheapest sell orders in the order book, while the opposite happens for selling trades.
One trader’s impact is another trader’s alpha.

A trader cannot directly observe two strategies’ outcomes for a single trade. We can do live trading experiments with two strategies A and B chosen randomly (A-B tests); or back test to compare strategies A and B on historical or simulated data.

* "Practitioners use impact models as a pre-trade tool to estimate the expected transaction cost of an order and to optimize the execution strategy."* Bershova and Rahklin (2013)

## Transaction Cost Analysis

Regular TCA reports enable traders to improve trading algorithm and find inconsistencies with back tests.

Unbiased A-B tests confirm that trading slower overall outperforms trading faster. Agressive strategies have more slippage but also trade when we have more alpha.

Some trading signal questions include whether a signal is directional (ex: return prediction) or non-directional (ex: volatility prediction)
What period of time are we intersted in? (next tick, next hour or at the day's close).
How frequently does a signal trigger? Conditioning the signal on Bloomberg news events on the stock may help to get an alpha that overcome the bid-ask spread.
