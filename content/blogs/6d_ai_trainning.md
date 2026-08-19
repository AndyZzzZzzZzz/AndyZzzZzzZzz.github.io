+++
title = "Modern 6D parallelism on distributed training"
date = ""
description = "Brief introduction on parallel training technique/strategy with modern large language model"
summary = ""
draft = true
+++

I remember the first time I hear about 3D training or 6D training during a meeting with my mentor, the first idea comes to my mind is watching 3D movie in theater. Fast forward to now, I wanted to share the initial note I wrote when doing research about this online.

Modern distributed AI training splits its workload in multiple dimensions across distributed clusters. Some of the reaons for this includes performance improvement and the fact that single cluster can't hold too much data. Some of the core challenges these strategy attempt to tackle:

1. The model's parameters (billion scale) is too large to fit in memory
2. The arithemetic performed on model tensor between layer to layer may be too slow on one or couple GPUs.
3. A single training stream (pipeline) may be too slow

Essentially 6D parallelism is an informal umbrella term for combining six complementary ways to devide that work:

1. Data parallelism (DP) / Fully Sharded Data parallel (FSDP)
2. Tensor parallelism (TP)
3. Pipeline parallelism (PP)
4. Seuqnece parallelism (SP)
5. Context parallelism (CP)
6. Expert parallelism (EP)

Note that I specifically labeled these in the above order for how common they are seen in pretraining phase of LLM (or other models). This means that you are more likely to see DP, TP, PP combined as 3D training compare to approaches like EP. These dimensions represent which part of the training problem we should distribute: training data, model tensors, layers, token positions, attention context, or experts.

## What a training step does (highlevel)

A training step usually has three phases:
- **forward pass**: tokens travel through the model to produce prediction
- **loss computation**: the model's predication are compared with the correct next tokens. The result number measures prediction error that needs to be lowered.
- **backward pass**: automatic differentiation computes gradients, which represent how each parameter should change to reduce the loss.


Its worth notice that a model's memory use is broader than just its weight. For a concrete memory example: 340B-parameter Nemotrol model.
