---
layout: post
title: Generative AI resources
---

* [Tokenizer playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)
* [Comparison of providers](https://artificialanalysis.ai/models/llama-3-instruct-70b/providers)

# LLM leaderboards:

* [OpenLLM](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
* [LMSYS Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
* [LLM Rubric Leaderboard](https://tide-freckle-52b.notion.site/1e0168e3481747ebaa365f77a3af3cc1?v=83e3d58d1c3c45ad879834981b8c2530)

# List of papers to filter:

* Efficient LLM papers on [github](https://github.com/tiingweii-shii/Awesome-Resource-Efficient-LLM-Papers?tab=readme-ov-file)
* LLM papers on [github](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file)
* Prompt engineering [guide](https://www.promptingguide.ai/research/rag#rag-research-insights) on RAG

# Courses:

* Maxime Labonne's [course](https://github.com/mlabonne/llm-course?tab=readme-ov-file)
* The novice LLM [training guide](https://rentry.org/llm-training)
* [Getting started](https://www.youtube.com/watch?v=nOxKexn3iBo&t=621s) with CUDA for Python programmers
* Princeton COS 597G (Fall 2022) on [youtube](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
* Stanford NLP processing with Deep Learning on [youtube](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)
* [General content](https://jalammar.github.io/) about AI
* Microsoft Gen AI [for beginners](https://microsoft.github.io/generative-ai-for-beginners/#/13-continued-learning/README?wt.mc_id=academic-105485-koreyst&id=lesson-6-building-text-generation-applications)

# Interesting blogs:

* [kipp.ly](https://kipp.ly/transformer-inference-arithmetic/) inference performance guide
* [blog euleuther](https://blog.eleuther.ai/transformer-math/) training guide
* [novice llm](https://rentry.org/llm-training) training guide
* [Llama Factory](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file) fine-tuning guide
* Llama [inference speed benchmarks](https://github.com/premAI-io/benchmarks)
* [answer.ai](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) on how to train a 70b model at home

Memory = 1.2 * Number of parameters * (precision / 8 bits)
For instance, Llama-3-70B in bfloat16 -> Memory required ~ 1.2 * 70B * 16/8 = 168GB ~ 2xA100 (80GB)

# Good ressources for GPU programming:

* nvidia cuda [training series](https://www.olcf.ornl.gov/cuda-training-series/)
* nvidia [blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) introducing cuda
* advanced pytorch [tutorials](https://pytorch.org/tutorials/advanced/cpp_extension.html)
* Heterogenous Parallel Programming on [youtube](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb)
* lecture on applied gpu programming on [youtube](https://www.youtube.com/playlist?list=PLPJwWVtf19Wgx_bupSDDSStSv-tOGGWRO)

# Papers (tool for researching papers: [arxiv-sanity](https://arxiv-sanity-lite.com/)  )

* https://github.com/hollobit/GenAI_LLM_timeline
* https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation#evaluation-leaderboards

Bunch of papers to read:
* Karpathy list of papers: https://twitter.com/karpathy/status/1734659057938477174
* [HuggingFace - Mixture of Experts](https://huggingface.co/blog/moe)
* [BloombergGPT](https://arxiv.org/pdf/2303.17564.pdf) (pretrained model according to Chinchilla law)
