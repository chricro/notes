---
layout: post
title: Large Language Models
---

## Introduction to Large Language Models

Here is a list of papers that give a good introduction to LLMs:

-[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) from Vaswani et al. (Google) | 06-2017 | transformer architecture and self-attention mechanism
-[An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929.pdf) from Dosovitskiy et al. (Google) | 10-2020 | ViT architecture (positional encoding added on patches)
-[LlaMa 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf) from Touvron et al. (FAIR) | 07-2023 | Helpfuness and safety alignment (RLHF)
-[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) from Hoffmann et al. (DeepMind) | 03-2022 | Chinchilla paper
-[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf) from Stanford | 05-2023 | different alignment algorithm (simpler than RLHF)

- [VOYAGER](https://arxiv.org/pdf/2305.16291.pdf): An Open-Ended Embodied Agent with Large Language Models  
Minecraft agent stocking knowledge -> how the knowledge is retrieved? simply cosine similarity? (01/12/2023)

To understand the novelty and power of transformers, one can read the following paper which shows some of the challenges using RNN architectures in NLP.

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) from Bahdanau et al. (2016). Improvement on RNN architecture to translate longer sequences (context parameter)

Since the official release of GPT-3.5, OpenAI's model has been far beyond both OSS and proprietary models. It is largely assumed that they are using a Mixture of Experts architecture. Mistral gained a significant attention in December 2023 by releasing Mixtral-8x7B which became the most efficient OSS model at that time, using a MoE architecture. MoE is not new in the research domain, here is a list of key papers on this subject:

- [Outrageously large neural networks: the sparsely-gated mixture-of-experts layer](https://arxiv.org/pdf/1701.06538.pdf) from Google (01-2017)
- [A review of sparse expert models in deep learning](https://arxiv.org/pdf/2209.01667.pdf) from Google (09-2022)
- [Megablocks: efficient sparse training with mixture of experts](https://arxiv.org/pdf/2211.15841.pdf) from Stanford, Microsoft, Google (11-2022)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/pdf/2202.09368.pdf) from Google (02-2022)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) from Google (01-2021)

Instead of having numerous experts (which are essentially FFN layers) during training/inference, another original technique called *model merging* focuses on merging the weights of different LMs directly.

* Phixtral MoE (Maxime Labonne): [mergekit](https://github.com/cg123/mergekit/tree/mixtral)
* [MoE for clowns](https://goddard.blog/posts/clown-moe/)
* Huggingface collection of [Model merging]([https://huggingface-co.translate.goog/collections/osanseviero/model-merging-65097893623330a3a51ead66?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=sc](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66))
* Linear and [Task Arithmetic](https://arxiv.org/pdf/2212.04089.pdf) merging
* Two MoE papers: https://arxiv.org/pdf/1701.06538.pdf and https://arxiv.org/pdf/1312.4314.pdf
* [gdoc to read](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit?pli=1)


## Speedup the inference of LLMs

LLMs are huge models (requiring often 7-8B parameters for small models; around 45-70B parameters for larger models and more than 1,700B for GPT-4 models). Therefore, the forward propagation requires the inputs to go through all the model weights neurons during inference to generate the next predicted tokens, which can take some time before getting the whole generated text.

We can be interested in optimizing the throughput (number of tokens generated per second, in batch size >=1) or the latency (number of tokens generated per second in batch size = 1).

For latency optimization, speculative decoding techniques are often relevant. These technique use a small *draft model* to predict large “target” model’s output, ultimately accelerating inference.

- [MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774.pdf) from TogetherAI (01-2024): only implemented with Vicuna models for now.
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077.pdf) from Microsoft (01-2024)

For throughput optimization, KV cache optimization is interesting:

- SGLang (RadixAttention): A crucial optimization in SGLang involves KV cache reuse. This approach allows prompts with identical prefixes to share intermediate KV cache, reducing redundant memory and computation.
- LightLLM introduces a more fine-grained KV cache management algorithm called TokenAttention and designs an Efficient Router scheduling implementation that works efficiently with TokenAttention. (https://github.com/ModelTC/lightllm/blob/main/docs/LightLLM.md)
- [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/pdf/2401.08671.pdf) from Microsoft (01-2024)

Here is a list of frameworks for LLM inference: https://llm.extractum.io/gpu-hostings/
Some popular frameworks for model deployment are vLLM, TensorRT (tensorrt-llm), Deepspeed, ONNX.

You can try out inference with different LLMs for free with [OpenRouter](https://openrouter.ai/) or [Perplexity](https://labs.perplexity.ai/), which gives a blazing-fast inference. They also provide interesting insights in their blog, such as an [A100/H100 comparison experiments](https://blog.perplexity.ai/blog/turbocharging-llama-2-70b-with-nvidia-h100).

To improve the throughput/latency, we can also leverage quantization techniques. Usually, these techniques enable people to use a model using less memory (which in some cases enable GPU poors to just train/infer a model).

A good comparison of different quantization techniques can be found in this [link](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/). Generally, AWQ < GPTQ.

- [FP8 vs INT8](https://arxiv.org/pdf/2303.17951.pdf)

In 2022, a relatively high generated output in batch size 1 would be around 150 tokens/s for medium models (45-70B params); and 300 tokens/s for small models (7-8B params). Groq achieves to attain 500 tokens/s for medium models (x2-3 on FireWorksAI, TogetherAI specialized in accelerated inference). This is just a trick, since they don't use classic GPUs (A100, H100) but their own TPU (Tensor Processing Unit), enabling them to get a deterministic control of the tokens generation.

- Paper on the [tensor streaming processor](https://wow.groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf) and the corresponding [youtube video](https://www.youtube.com/watch?v=xTT2GpdSRKs)
- [GroqDay](https://www.youtube.com/watch?v=upljocX5mrk) in December 2021
- [papers](https://wow.groq.com/category/papers/) recommended by groq
- all these resources were found on this [reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1auxm3q/comment/krb3twr/?utm_source=share&utm_medium=web2x&context=3)


## Improve the reasoning of LLMs

Reasoning and planning approaches:

- [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406.pdf)
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465.pdf)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/pdf/2403.09629.pdf)

To understand the transformers capabilities, one can perform *prompt-engineering*:

- https://www.promptingguide.ai/introduction/tips
- https://lojones.github.io/2023/04/30/asimov-prompt-engineer.html (Asimov)

An important point is the *context-window* of models to keep in context large conversations and every important element:

- https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c

## Transformers aspects to improve:

Positional encoding introduced by Transformers (2017): learned absolute positional encoding with sine and cosine positional functions.
Other techniques involve learned relative positional encoding. Rotary Position Embedding (RoPE): https://blog.eleuther.ai/yarn/  
[Shikra](https://arxiv.org/pdf/2306.15195.pdf): 6.2 Location tokens or just numbers?

Flash-decoding for long-context inference: https://crfm.stanford.edu/2023/10/12/flashdecoding.html  
Flash attention brings down the computation time of attention from quadratic to linear (https://arxiv.org/pdf/2205.14135.pdf), explained here: https://shreyansh26.github.io/post/2023-03-26_flash-attention/

## Multi-modality

Some insights on GPT4-V model on image processing is described in this: https://platform.openai.com/docs/guides/vision

Some of the interesting OSS ViTs (in late 2023) are CogVLM, Adept Fuyu, MiniGPT, BLIP-2, LlaVa-1.5.

We often find parallel between vision models and software agents, since vision on desktop can be something really helpful.

- [CogAgent](https://arxiv.org/pdf/2312.08914.pdf) is trained on desktop images.
- [Mind2Web](https://arxiv.org/pdf/2306.06070.pdf) is a WebAgent.
- Retrieve information: https://arxiv.org/pdf/2005.11401.pdf
- LLM solving computer tasks: https://arxiv.org/pdf/2303.17491.pdf

Other techniques try to combine LLMs and browser softwares, such as GPT4-V Act, which uses vision with CSSOM (CSS version of DOM) https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect , https://github.com/ddupont808/GPT-4V-Act/issues/4

A nice summary of Vision-Language models is provided in a [paper](https://arxiv.org/pdf/2405.17247) from Meta (24/05/2024)

Among popular models in 2023, we can find:

LLaMa series (2023-02), up to 65B parameters and 2,048 context-window. 3 modifications compared to the original transformer architecture (RMSNorm for pre-normalization; rotary embeddings; SwiGLU activation function). Then the LLaMa2 series (2023-07) up to 70B and 4,096 context-window focused on optimizing dialogue use case. As a comparison, GPT4 (2023-03) has 1,760B parameters and 8,192 and 32,768 context-window. GPT4 generated data are often use to fine-tune other models, such as Orca2 (a descendant of LLaMa2 developed by Microsoft in 2023) which trains LLaMa2 base models on high-quality synthetic data from GPT-4. 

See the [LLM index](https://sapling.ai/llm/index?WT.mc_id=academic-105485-koreyst) for more model comparison

Chinchilla law: number of tokens during training (dataset size including epochs) should be ~20x the number of parameters of the model


# Random ML stuff

Recall = TP / (TP + FN) : true predictions compared to the positive  
Precision = TP / (TP + FP) : true predictions compared to the predicted positive
