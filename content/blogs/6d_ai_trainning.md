+++
title = "Modern 6D parallelism on distributed training"
date = '2026-08-20T17:21:37+08:00'
description = "A brief introduction to parallel training techniques and strategies for modern large language models"
summary = "A practical breakdown of modern 6D parallelism techniques (DP/FSDP, TP, PP, SP, CP, and EP) for training large language models, including a complete 128-GPU cluster assignment example."
draft = false
+++

I remember the first time I heard about 3D or 6D training during a meeting with my mentor—the first thing that came to my mind was watching a 3D movie in a theater. Fast forward to now, I wanted to share the initial notes I wrote while researching this topic online.

Modern distributed AI training splits its workload across multiple dimensions in distributed clusters. Some of the primary reasons for this include performance improvements and the simple fact that a single GPU node cannot hold that much data or model state. Some of the core challenges these strategies attempt to tackle are:

1.  The model's parameters (on a billion-plus scale) are too large to fit in a single GPU's memory.
2.  The arithmetic performed on model tensors from layer to layer may be too slow on just one or a couple of GPUs.
3.  A single training stream (pipeline) may be too slow to meet throughput requirements.

Essentially, 6D parallelism is an informal umbrella term for combining six complementary ways to divide that work:

1. Data parallelism (DP) / Fully Sharded Data parallel (FSDP)
2. Tensor parallelism (TP)
3. Pipeline parallelism (PP)
4. Sequence parallelism (SP)
5. Context parallelism (CP)
6. Expert parallelism (EP)

Note that I specifically listed these in the order of how commonly they are seen in the pretraining phase of LLMs (and other models). This means you are much more likely to see DP, TP, and PP combined as "3D training" compared to specialized approaches like EP. These dimensions represent which part of the training workload we choose to distribute: training data, model tensors, layers, token positions, attention context, or experts.

## What a training step does (high level)

A training step usually has three phases:
- **forward pass**: tokens travel through the model to produce prediction
- **loss computation**: the model's predictions are compared with the correct next tokens. The result number measures prediction error that needs to be lowered.
- **backward pass**: automatic differentiation computes gradients, which represent how each parameter should change to reduce the loss.


It is worth noting that a model's memory footprint goes far beyond just its weights. The following components consume VRAM during training:

- **parameter (weights)** essentially the learned numerical values 
- **gradients** parameter-sized values produced during backpropagation
- **optimizer state** extra values maintained by optimizer (this can consume multiple times the parameter memory)
- **activations** intermediate layer outputs saved for the backward pass. (dominate memory for long sequences)
- **communication buffers** sending tensors between GPUs

For a rough estimate, training a dense 340-billion parameter model (with BF16 parameters/gradients and FP32 master weights) will incur anywhere between 5TB to 6TB of total memory usage across the cluster during training.

## Data Parallelism (DP)

Traditional DP makes full copies of a model and gives each copy a different portion of the global training batch. A "batch" here represents the total collection of data used for a single optimizer update step. If the global batch size is 1,024 sequences and there are 8 DP GPUs, each GPU processes 128 unique sequences using its own complete replica of the model.

1. every gpu begins with identical model parameters
2. each gpu runs forward and backward passes on its local mini-batch
3. each gpu has gradients for the same parameters, but from different examples
4. the gpus use an `all-reduce` operation to sum or average those gradients
5. each gpu applies the same optimizer update, keeping the model copies identical

This traditional approach fails for modern frontier models because it requires every GPU to hold a full copy of the model parameters, gradients, and optimizer states. If the model state alone requires 200GB but your GPU only has 80GB of VRAM, a single replica cannot even initialize.

## FSDP: data parallelism with model state sharded across ranks

Fully Sharded Data Parallelism (FSDP) maintains data-parallel training semantics, but shards the persistent model state across data-parallel groups. A shard is a non-overlapping slice of a tensor. Rather than storing a full parameter vector on every GPU, each GPU permanently owns only a portion of it. FSDP commonly shards parameters, gradients, and optimizer states.

For example, consider a model distributed over 4 FSDP ranks (4 GPUs):
-   Rank 0 owns a shard (25%) of the parameters, gradients, and optimizer states.
-   Rank 1 owns a shard (25%) of the parameters, gradients, and optimizer states.
-   Rank 2 owns a shard (25%) of the parameters, gradients, and optimizer states.
-   Rank 3 owns a shard (25%) of the parameters, gradients, and optimizer states.

When executing the model:
-   Before computing a specific layer, the ranks in the relevant FSDP group execute an **all-gather** operation to temporarily reconstruct the full parameters needed for that layer.
-   Each rank then computes that layer on its own unique batch of data.
-   Once the layer computation finishes, the gathered full parameters are discarded from memory to save VRAM.

Here, an **all-gather** assembles shards contributed by all ranks so each participant obtains the full tensor. Essentially, it grabs missing pieces from everyone and hands the complete result back to all participants, ensuring every rank is on the same page.

At a high level, FSDP is still a form of data parallelism because ranks process different data examples. However, persistent weights are not held as full copies on every replica. Instead, the GPUs work as a team, with each rank holding a slice of the weights, gradients, and optimizer states in memory.

In a larger cluster setup (e.g., 16 GPUs), we can form multiple FSDP groups. For instance, we might form 4 groups of 4 ranks each, resulting in 4 model replicas across the cluster:
- **Replica 0**: (rank 0, rank 1, rank 2, rank 3)
- **Replica 1**: (rank 4, rank 5, rank 6, rank 7)
- **Replica 2**: (rank 8, rank 9, rank 10, rank 11)
- **Replica 3**: (rank 12, rank 13, rank 14, rank 15)

FSDP lowers memory for GPU, but instead increases communication. Now parameters must be gathered when needed and gradients scattered after use. Good performance therefore depends on fast interconnects, overlap of communication with computation, and careful wrapping or sharding boundaries.

## Tensor Parallelism (TP): split one layer's matrix math

TP divides the arithmetic and parameters inside an individual neural-network layer across several GPUs. It is used when a single layer is too large or too expensive for one GPU. tensor is essentially a multi-dimensional array (a 1D tensor is a vector, and a 2D tensor is a matrix). Neural-network weights are often 2d tensors. A transformer's linear layer can be written as:

```
Y = XW
```
Here, `X` is an input activation matrix. `W` is a learned weight matrix, and `Y` is the output. The dominant operation is **matrix multiplication** (multiplying rows of `X` by columns of `W`)

### Column parallelism (vertical split)

The first way to distribute the workload is by cloumns:
```
W = [ W0 | W1 ]
Y = [ XW0 | XW1 ]
```
Here, rank 0 to store `W0` and computes `XW0`, rank 1 to store `W1` and computes `XW1`. Their outputs are concatenated. 

### Row parallelism (horizontal split)

The second way is to split `W` by rows:
```
W = [ W0
      W1 ]
X = [ X0 | X1 ]
Y = X0W0 + X1W1
```
We use rank 0 to compute `X0W0` and rank 1 to compute `X1W1`. The partial outputs must be summed, commonly via an **all-reduce** operation.

If the job has 8 ranks and `TP=2`, there are 4 TP groups available to perform computation:
-   **TP Group 0**: Rank 0, Rank 1
-   **TP Group 1**: Rank 2, Rank 3
-   **TP Group 2**: Rank 4, Rank 5
-   **TP Group 3**: Rank 6, Rank 7

If we combine this with Data Parallelism (TP=2, DP=4), then every TP group holds a copy of the model (with each layer's matrices split 50/50 between two GPUs):
-   **TP Group 0**: process data batch A
-   **TP Group 1**: process data batch B
-   **TP Group 2**: process data batch C
-   **TP Group 3**: process data batch D

Due to the need to exchange or combine data after computation, TP groups usually stays within a node or tightly coupled GPU island. At Nvidia, we often use **NVLink** (high-bandwidth GPU interconnect) for local TP groups. TP's frequent collectives are why a fast local fabric matters.

## Pipeline Parallelism (PP): split the model by depth

PP partitions the model into consecutive groups of layers called **pipeline stages**. Each stage is often placed on a different GPU group or server.

For a 30 layer model with three stages:
- Stage 0 / Server A hosts layer 1-10
- Stage 1 / Server B hosts layer 11-20
- Stage 2 / Server C hosts layer 21-30

Just like assembly line workers in a factory, the activations produced by one stage are sent to the next stage. No stage needs to store every layer. This idea is similar to a CPU pipeline, where the work is separated by stages to improve the overall throughput.

PP also splits one batch into many micro-batches and streams them through the stages. Without micro-batching, stage 1 and stage 2 wait while stage 0 processes the entire batch. Essentially the idea is that we want to pump work little by little into each stage while keeping everyone busy.
```
Time →       1       2       3       4       5
Stage 0      μ1      μ2      μ3      μ4      μ5
Stage 1              μ1      μ2      μ3      μ4
Stage 2                      μ1      μ2      μ3
```

An example of the most common LLM pretaining (3D Parallelism):

Let's use 16 GPUs (ranks 0 - 15) with the following setup (PP=2, TP=2, DP=4) to train a model with 32 layers. This means we split the 32 layers depth-wise into 2 stages, split each layer's matrix math between 2 GPUs, and create 4 data processing replicas across the cluster.

-   **PP Stage 0** (layers 1-16)
    -   **DP 0** replica 0: **TP 0** rank 0 + rank 1
    -   **DP 1** replica 1: **TP 1** rank 2 + rank 3
    -   **DP 2** replica 2: **TP 2** rank 4 + rank 5
    -   **DP 3** replica 3: **TP 3** rank 6 + rank 7
-   **PP stage 1** (layers 17-32)
    -   **DP 0** replica 0: **TP 0** rank 8 + rank 9
    -   **DP 1** replica 1: **TP 1** rank 10 + rank 11
    -   **DP 2** replica 2: **TP 2** rank 12 + rank 13
    -   **DP 3** replica 3: **TP 3** rank 14 + rank 15

How a data batch flows through this system
1.  Four different batch of data enters
    -   replica 0 receives data batch A
    -   replica 1 receives data batch B
    -   replica 2 receives data batch C
    -   replica 3 receives data batch D
2.  First half of the model runs (PP stage 0 + TP = 2)
    -   For batch A, ranks 0 and 1 work toegther to solve layer 1-16
    -   Simultaneously, ranks 2 & 3 process batch B, ranks 4 & 5 process batch C etc
3.  Activations are pipelined across servers
    -   Once PP stage 0 finishes layers 1-16 for a micro-batch, ranks (0,1) send their output activations across the network to ranks (8,9)
4.  Second half of the model runs
    -   Ranks (8, 9) work together to compute layer 17-32 on batch A, calculate the loss, and begin the backward pass.

## Sequence Parallelism (SP): split token-wise activation work

A language model processes a sequence: an ordered list of tokens. A **token** is a chunk of text processed by the model (it can be a word, word fragment, punctuation mark, or code symbol).

SP splits parts of the computation that operate independently across token positions. It is most commonly paired with TP and is designed to reduce duplicate activation memory within a tensor-parallel group. During TP, GPUs split up matrix math. But in a Transformer, there are also "non-matrix" layers like **LayerNorm** and **Dropout**:
- **LayerNorm** is a stabilization technique that keeps the numbers moving through deep neural nets stable.
- **Dropout** is a regularizer that prevents the model from memorizing the training data (overfitting).

Both of these techniques are element-wise or token-wise operations. Under plain TP, every GPU holds duplicate activation memory for the entire sequence during these steps. This means if our sequence is 4,000 tokens long, storing all 4,000 tokens on every GPU causes a massive waste of VRAM.

Instead of duplicating the whole sequence across every GPU:

1.  Rank 0 takes tokens 1–2,000 and rank 1 takes tokens 2,001–4,000.
2.  Each GPU runs LayerNorm or Dropout on its own assigned tokens.
3.  The GPUs share their tokens right before hitting a heavy matrix multiplication step (TP layer), do the math, and split them back up right after.

The approach is primarily targeting memory optimization when training large language models.


## Context Parallelism (CP): split long-context attention

CP divides the sequence/context dimension across GPUs for attention computation. It becomes especially important for tasks involving very long prompts or documents.

In self-attention, each token can use information from other tokens in the context. The model derives three learned representations:
- **Query (Q):** what this token is looking for
- **Key (K):** what this token offers as an address or label
- **Value (V):** the information to retrieve if the key is relevant

The core memory challenge is that attention compares every token to every other token. With 1,000 tokens, it takes 1 million operations. If you scale up to 128,000 tokens in a long-context LLM, the memory easily explodes.

Context Parallelism solves this by slicing the long input document across multiple GPUs specifically during the Attention calculation.

If you have a 100,000-token prompt and 4 GPUs with `CP=4`:

1. **Slice the Tokens:** Each GPU takes a 25,000-token chunk of the prompt.
    - GPU 0 holds $Q_0, K_0, V_0$ (Tokens 1–25,000)
    - GPU 1 holds $Q_1, K_1, V_1$ (Tokens 25,001–50,000)
    - GPU 2 holds $Q_2, K_2, V_2$ (Tokens 50,001–75,000)
    - GPU 3 holds $Q_3, K_3, V_3$ (Tokens 75,001–100,000)
2. **Ring Exchange (Ring-Attention):** To calculate full attention, GPU 0 needs its Queries to see the Keys and Values from all 100,000 tokens.
    - In step 1, GPU 0 calculates attention between $Q_0$ with its local $K_0$, $V_0$.
    - In step 2, GPU 0 sends its $K_0$, $V_0$ block to GPU1, while receiving $K_3$, $V_3$ from GPU 3.
    - It calculates partial attention with this new block, updates its running scores, and passes the K, V block down the line again repeating this process.
3.  The K, V block pass around the GPUs in a ring until every GPU's Queries have seen every single Key and Value in the 100,000-token sequence.

CP makes much longer contexts practical, but it requires substantial communication and careful overlap. It is most valuable when context length is the main memory or compute bottleneck.

## Expert Parallelism (EP): distribute mixture-of-experts layers

MoE is a newer approach that is becoming extremely popular in modern LLM training, especially when scaling up a dense model's size hits diminishing returns. It replaces the standard dense feed-forward subnetwork in a layer with many specialized subnetworks called **experts**. A learned **router** (or gate) selects a small number of experts for each token.

For example, an MoE layer might have 64 experts but route each token to only its top 2 experts. The layer has far more total parameters than a dense layer, while the compute per token remains low—closer to running just two experts rather than all 64. Experts specialize according to patterns the training objective discovers, which may be difficult to name or may correspond to syntactic, lexical, multilingual, or other distributed behaviors rather than clean topics.

EP places different experts on different GPUs. A token may begin on GPU 0, be routed to an expert sitting on GPU 1, and have its output returned to GPU 0 for the next part of the model. This requires an **all-to-all** style exchange: ranks send different token subsets to different destinations based on where the router assigns them.

## Combining the dimensions

Concrete example for 6D Parallelism example with FSDP using a realistic cluster setup of 128 GPUs (rank 0-127) training a 64 layers MoE LLM.

```text
Total GPUs = PP × EP × CP × TP × FSDP = 2 × 2 × 2 × 2 × 4 = 128 GPUs
```

1.  PP=2, splits the model's 64 total layers in half
    -   PP stage 0: layers 1-32 (executed on 64 GPUs)
    -   PP stage 1: layers 33-64 (executed on 64 GPUs)
2.  EP=2, inside each MoE layer (8 experts), the 8 experts are split across 2 GPU groups
    -   EP group 0: hold experts 1-4
    -   EP group 1: hold experts 5-8
3.  CP=2, slices long prompt (64,000 tokens) in half for attention
    -   chunk 0: tokens 1-32,000
    -   chunk 1: tokens 32,001-64,000
4.  SP, integrated directly into the TP groups to shard activation memory during LayerNorm and Dropout steps
5.  TP=2, splits internal linear weight matrices between 2 local GPUs using NVLinks
6.  FSDP=4, creates 4 data parallel model replicas. Inside each replica group of 4 GPUs, the persistent model parameters, gradients, and optimizer states are sharded 4 ways (25% per GPU)

Here is how rank 0-15 are assigned within PP stage 0 (layer 1-32)
| Rank | FSDP Group | Context Chunk (CP) | Expert Group (EP) | Tensor/Sequence Role (TP/SP) |
| :--- | :--- | :--- | :--- | :--- |
| **Rank 0** | **FSDP Group 0** (Shard 0) | Chunk 0 (Tokens 1–32k) | Experts 1–4 | Matrix Split 0 (50%) |
| **Rank 1** | **FSDP Group 1** (Shard 0) | Chunk 0 (Tokens 1–32k) | Experts 1–4 | Matrix Split 1 (50%) |
| **Rank 2** | **FSDP Group 0** (Shard 1) | Chunk 1 (Tokens 32k–64k) | Experts 1–4 | Matrix Split 0 (50%) |
| **Rank 3** | **FSDP Group 1** (Shard 1) | Chunk 1 (Tokens 32k–64k) | Experts 1–4 | Matrix Split 1 (50%) |
| **Rank 4** | **FSDP Group 0** (Shard 2) | Chunk 0 (Tokens 1–32k) | Experts 1–4 | Matrix Split 0 (50%) |
| **Rank 5** | **FSDP Group 1** (Shard 2) | Chunk 0 (Tokens 1–32k) | Experts 1–4 | Matrix Split 1 (50%) |
| **Rank 6** | **FSDP Group 0** (Shard 3) | Chunk 1 (Tokens 32k–64k) | Experts 1–4 | Matrix Split 0 (50%) |
| **Rank 7** | **FSDP Group 1** (Shard 3) | Chunk 1 (Tokens 32k–64k) | Experts 1–4 | Matrix Split 1 (50%) |
| ... | ... | ... | ... | ... |

Ranks 8–15 mirror Ranks 0–7, but hold EP Group 1 (Experts 5–8) instead of Experts 1–4. Together, Ranks 0–15 form 1 complete model replica for PP Stage 0.

Ranks 16–63 duplicate this 16-rank structure 3 more times to create Data Parallel Replicas 1, 2, and 3 for PP Stage 0.

Ranks 64–127 mirror the exact same 64-GPU layout for PP Stage 1 (Layers 33–64).

How does data flow through (rank 0's perspective):
1. **Persistent Memory State (FSDP):** Rank 0 permanently holds only 25% (a shard) of the model weights, gradients, and optimizer states assigned to FSDP Group 0.
2. **Materializing Weights (FSDP All-Gather):** Before running Layer 1, Rank 0 communicates with Ranks 2, 4, and 6 via `all-gather` to temporarily build the full weights for Matrix Split 0 of Layer 1 in VRAM.
3. **LayerNorm / Dropout (SP):** Rank 0 and Rank 1 split the 32,000 tokens of Chunk 0. Rank 0 runs LayerNorm independently on tokens 1–16,000.
4. **Attention Computation (CP + TP):**
   - **TP:** Rank 0 and Rank 1 use an `all-gather` to collect all 32,000 tokens of Chunk 0 and split the QKV matrix multiplication 50/50.
   - **CP:** Rank 0 rotates $K, V$ blocks in a ring with Rank 2 so its Queries see all 64,000 tokens in the prompt.
5. **Routing Tokens (EP All-to-All):** The model hits the MoE sub-layer. The router assigns tokens to experts:
   - Tokens assigned to **Experts 1–4** stay within Ranks 0–7.
   - Tokens assigned to **Experts 5–8** are sent across the network via an `all-to-all` exchange to **Ranks 8–15**.
6. **Freeing Memory (FSDP Drop):** Once Layer 1 finishes computation, Rank 0 immediately discards the temporary full Layer 1 weights from VRAM, keeping only its assigned 25% persistent shard before moving to Layer 2.




