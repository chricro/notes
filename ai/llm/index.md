---
layout: post
title: Large Language Models
---

Phixtral MoE (Maxime Labonne): [mergekit](https://github.com/cg123/mergekit/tree/mixtral), [MoE for clowns](https://goddard.blog/posts/clown-moe/)
Huggingface collection of [Model merging]([https://huggingface-co.translate.goog/collections/osanseviero/model-merging-65097893623330a3a51ead66?_x_tr_sl=en&_x_tr_tl=fr&_x_tr_hl=fr&_x_tr_pto=sc](https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66))
Linear and [Task Arithmetic](https://arxiv.org/pdf/2212.04089.pdf) merging
Two MoE papers: https://arxiv.org/pdf/1701.06538.pdf and https://arxiv.org/pdf/1312.4314.pdf

# Groq
* paper on the [tensor streaming processor](https://wow.groq.com/wp-content/uploads/2020/06/ISCA-TSP.pdf) and the corresponding [youtube video](https://www.youtube.com/watch?v=xTT2GpdSRKs)
* [groqday](https://www.youtube.com/watch?v=upljocX5mrk) in december 2021
* [papers](https://wow.groq.com/category/papers/) recommended by groq
* all these resources were found on this [reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1auxm3q/comment/krb3twr/?utm_source=share&utm_medium=web2x&context=3)

## MoE papers

- [Outrageously large neural networks: the sparsely-gated mixture-of-experts layer](https://arxiv.org/pdf/1701.06538.pdf) from Google (01-2017)
- [A review of sparse expert models in deep learning](https://arxiv.org/pdf/2209.01667.pdf) from Google (09-2022)
- [Megablocks: efficient sparse training with mixture of experts](https://arxiv.org/pdf/2211.15841.pdf) from Stanford, Microsoft, Google (11-2022)
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/pdf/2202.09368.pdf) from Google (02-2022)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf) from Google (01-2021)

## Speedup the inference of LLMs

- [MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774.pdf) from TogetherAI (01-2024)
- [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/pdf/2401.08671.pdf) from Microsoft (01-2024)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077.pdf) from Microsoft (01-2024)

List of LLM inference: https://llm.extractum.io/gpu-hostings/

vLLM; TensorRT on Mixtral 8x7b: coming soon on (only Llama for now)
* https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#mistral-v01
* https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/mixtral/README.md
* https://docs.mistral.ai/self-deployment/trtllm/ \
Deepspeed: coming soon on Mixtral 8x7b
SGLang: RadixAttention https://lmsys.org/blog/2024-01-17-sglang/, only available with A10G
ONNX (convert the model to a graph for optimization): not yet available with Mixtral (the exportation to ONNX format)
Medusa layers (only Vicuna for now)

Perplexity:
* [Introduction](https://blog.perplexity.ai/blog/introducing-pplx-api)
* [A100/H100 comparison experiments](https://blog.perplexity.ai/blog/turbocharging-llama-2-70b-with-nvidia-h100)

AWQ < GPTQ generally
Comparison of different quantization techniques: https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/

[gdoc to read](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit?pli=1)
[FP8 vs INT8](https://arxiv.org/pdf/2303.17951.pdf)

Optimising throughput:
* SGLang (RadixAttention): A crucial optimization in SGLang involves KV cache reuse. This approach allows prompts with identical prefixes to share intermediate KV cache, reducing redundant memory and computation.
* LightLLM introduces a more fine-grained KV cache management algorithm called TokenAttention and designs an Efficient Router scheduling implementation that works efficiently with TokenAttention. (https://github.com/ModelTC/lightllm/blob/main/docs/LightLLM.md)

Optimising latency:
* Speculative Decoding: Accelerate LLMs using a small “draft” model to predict large “target” model’s output

## DPO training

Experiments of 09/04/2024 and 10/04/2024 ([the code can be found here](https://github.com/chricro/personal-notes/blob/main/experiments/10-04-2024)):
DPO with HuggingFace Trainer: for large prompts, use gradient accumulating = 4; lower the learning rate (5e-7) since the complexity of the task; and increase beta to optimize the positive reward parameter. Don't train over too many data.


## Reasoning and planning approaches

- [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/pdf/2310.04406.pdf)
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465.pdf)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/pdf/2403.09629.pdf)

## Important papers

### LLM


-[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) from Vaswani et al. (Google) | 06-2017 | transformer architecture and self-attention mechanism
-[An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929.pdf) from Dosovitskiy et al. (Google) | 10-2020 | ViT architecture (positional encoding added on patches)
-[LlaMa 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf) from Touvron et al. (FAIR) | 07-2023 | Helpfuness and safety alignment (RLHF)
-[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) from Hoffmann et al. (DeepMind) | 03-2022 | Chinchilla paper |
-[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf) from Stanford | 05-2023 | different alignment algorithm (simpler than RLHF)

[VOYAGER](https://arxiv.org/pdf/2305.16291.pdf): An Open-Ended Embodied Agent with Large Language Models  
Minecraft agent stocking knowledge -> how the knowledge is retrieved? simply cosine similarity? (01/12/2023)


## Deprecated papers (but still interesting to read)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) | Bahdanau et al. | 2016 | Improvement on RNN architecture to translate longer sequences (context parameter)

# Other great resources

## Transformers aspects to improve:

Positional encoding introduced by Transformers (2017): learned absolute positional encoding with sine and cosine positional functions.
Other techniques involve learned relative positional encoding. Rotary Position Embedding (RoPE): https://blog.eleuther.ai/yarn/  
[Shikra](https://arxiv.org/pdf/2306.15195.pdf): 6.2 Location tokens or just numbers?

Flash-decoding for long-context inference: https://crfm.stanford.edu/2023/10/12/flashdecoding.html  
Flash attention brings down the computation time of attention from quadratic to linear (https://arxiv.org/pdf/2205.14135.pdf), explained here: https://shreyansh26.github.io/post/2023-03-26_flash-attention/

## GPT architecture

GPT4-V image processing: https://platform.openai.com/docs/guides/vision

## To understand the transformers capabilities

* Co-attention for VQA: https://proceedings.neurips.cc/paper_files/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf

* Prompt-engineering:
https://www.promptingguide.ai/introduction/tips
https://lojones.github.io/2023/04/30/asimov-prompt-engineer.html (Asimov)

* Context-window: https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c

## Interesting Language to vision models

GPT4-V Act: vision with CSSOM (CSS version of DOM) https://developer.mozilla.org/en-US/docs/Web/API/Element/getBoundingClientRect

https://github.com/ddupont808/GPT-4V-Act/issues/4

CogVLM, Adept Fuyu, MiniGPT, BLIP-2, ...  
[CogAgent](https://arxiv.org/pdf/2312.08914.pdf)

[Paper from Meta on Vision-Language models](https://arxiv.org/pdf/2405.17247)

Pretrained models:

- LlaMa | 2023-02 | Up to 65B | 2048 | Three modifications compared to the original transformer architecture: RMSNorm for pre-normalization; rotary embeddings; SwiGLU activation function
- LlaMa2 | 2023-07 | Up to 70B | 4096 | Fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases |
- GPT4 | 2023-03 | 1,760B | 8,192 and 32,768 | blabla |
- Orca 2 | 2023 | 7/13B | 4,096 | Descendant of LLaMA 2 developed by Microsoft obtained by fine-tuning the corresponding LlaMa 2 base models on tailored, high quality synthetic data from GPT4

see the [LLM index](https://sapling.ai/llm/index?WT.mc_id=academic-105485-koreyst) for more model comparison
Other benchmarks: Glue; SuperGlue; MMLU; bigbench, [elo dataset](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

# Random ML stuff

Recall = TP / (TP + FN) : true predictions compared to the positive  
Precision = TP / (TP + FP) : true predictions compared to the predicted positive

Chinchilla law: number of tokens during training (dataset size including epochs) should be ~20x the number of parameters of the model

## Vision-language: CogVLM

Retrieve information: https://arxiv.org/pdf/2005.11401.pdf
LLM solving computer tasks: https://arxiv.org/pdf/2303.17491.pdf

Try the different LLM at [OpenRouter](https://openrouter.ai/)

## WebAgent

[Mind2Web](https://arxiv.org/pdf/2306.06070.pdf)
