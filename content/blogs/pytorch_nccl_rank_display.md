+++
title = "Fixing a Stale Rank Bug in PyTorch's NCCL BackendT"
date = '2026-08-15T16:23:12+08:00'
draft = true
+++

Debugging distributed training in PyTorch can be challenging. When a collective operation times out or fails, your first instinct is to check the logs to see which rank caused the bottleneck. But what happens when the logs lie to you?

Recently, I encountered and patched an interesting bug in PyTorch's distributed backend (Issue#191305) where `WorkNCCL` error logs would report the incorrect local rank for processes participating in multiple process groups. Here's a blog documenting on why this happened, the investigation step, and the fix.

## Ambiguous Timeout Logs

In complex distributed topologies (like 3D parallelism), a single PyTorch process often belongs to multiple overlapping NCCL process groups. For example, a process might be part of a tensor parallel group, a pipeline parallel group, and a data parallel group.

Because of this, a process has a global rank (ID across the entire cluster) and a local rank (ID within a specific process group).

I noticed that when a process was part of multiple groups, PyTorch's NCCL watchdog would sometimes report the wrong local rank during a timeout. To isolate the issue, I wrote a scripte with three GPUs and two overlapping groups:
- Group A: global ranks `[0,1]`
- Group B: global ranks `[1,2]`

For the process with global rank 1 (position 1), its local rank in Group A is 1, but its local rank in Group B is 0 (position 0). When I intentionally triggered a timeout in Group A, the log correctly reported:

```plaintext
[global rank 1] timing out group_a; expected WorkNCCL prefix rank = 1
[Rank 1] Watchdog caught collective operation timeout: ...
```

However, when I triggered a timeout in Group B, the log output was incorrect:

```
[global rank 1] timing out group_b; expected WorkNCCL prefix rank = 0
[Rank 1] Watchdog caught collective operation timeout: ...
```

The expected prefix was `[Rank 0]`, but the log persistently claimed `[Rank 1] `. The log was caching the local rank from the first process group and blindly reusing it for all subsequent process groups.

## Root Cause Analysis

Diving into PyTorch C++ source code (`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`), we quickly found the culprit.

When PyTorch generates the log prefix for a NCCL work item, it calls `ProcessGroupNCCL::WorkNCCL::logPrefix()`. Here is what that function looked like:
```cpp
const std::string& ProcessGroupNCCL::WorkNCCL::logPrefix() const {
    static std::string prefix = c10::str("[Rank ", rank_, "] ");
    return prefix;
}
```

In C++, a function-local static variable is initialized exactly once during the first time the control flow passes theough its declaration.

When the first `WorkNCCL` instance (belonging to Group A) called `logPrefix()`, the static `prefix` string was initialized to `"[Rank A]"`. Because is was static, this variable lived for the lifetime of the program. When a completely different `WorkNCCL` instance (belonging to Group B with local rank 0) called `logPrefix()`, the initialization was skipped and the cached string was returned.

In a multi-tenant environment where a single process juggles multiple local ranks, this static caching breaks log accuracy.

## The Code Change
Instead of computing the prefix on the fly inside `logPrefix()`, we now construct a member variable `logPrefix_` inside the `WorkNCCL` constructor.

```cpp
// Inside the WorkNCCL constructor
if (pgDesc_.empty() || pgDesc_ == "undefined") {
  logPrefix_ = c10::str("[PG GUID ", pgUID_, " Rank ", rank_, "] ");
} else {
  logPrefix_ =
      c10::str("[PG GUID ", pgUID_, "(", pgDesc_, ") Rank ", rank_, "] ");
}
```

Then, the logging function simply returns the instance-specific string:
```cpp
const std::string& ProcessGroupNCCL::WorkNCCL::logPrefix() const {
  return logPrefix_;
}
```

## The Result
After compiling the patch, running the same reproducer yields perfectly accurate logs. Not only does the local rank update correctly, but the logs now explicitly tell you which process group is timing out via the `PG GUID`:

```plaintext
[global rank 1] timing out group_a; expected WorkNCCL prefix rank = 1
[PG GUID 1 Rank 1] Watchdog caught collective operation timeout: ...

[global rank 1] timing out group_b; expected WorkNCCL prefix rank = 0
[PG GUID 2 Rank 0] Watchdog caught collective operation timeout: ...
```

C++ unit test was also updated to prevent regression on this issue.

## Thoughts

I think this bus is a good example of how minor performance optimization can create significant observvability blind spots as a system's architecture evolves. What was likely a safe assumption in the early days of simple Data Parallelism broke down in the modern era of overlapping, multi-dimentional process groups.

## Notes

PG GUiD stands for Process Group Globally Unique Identifier. Every time the code calls `dist.init_process_group()` or `dist.new_group()`, PyTorch creates a new communication channel and assigns it an internal, unique integer ID (the GUID).

Quick reference table on the 3D parallel strategies

strategy, what it does, who is in the group
Data parallel (DP), Processes different batches of training data simultaneously. GPUs holding the same layer of the model 

Tensor Parallel (TP), Splits the math of a single matrix multiplication across GPUs, GPUs inside the same physical server 

Pipeline Parallel (TP), splits the model layer-by-layer, GPUs passing outputs sequentially to the next layer


The code I used to reproduced the error:
```cpp
  from datetime import timedelta

  import torch
  import torch.distributed as dist


  def main():
      dist.init_process_group("gloo")
      rank = dist.get_rank()
      torch.cuda.set_device(rank)

      timeout = timedelta(seconds=3)
      group_a = dist.new_group([0, 1], backend="nccl", timeout=timeout)
      group_b = dist.new_group([1, 2], backend="nccl", timeout=timeout)
      groups = [
          ("group_a", [0, 1], group_a),
          ("group_b", [1, 2], group_b),
      ]

      tensor = torch.ones(1, device="cuda")
      for name, ranks, group in groups:
          if rank in ranks:
              dist.all_reduce(tensor, group=group, async_op=True).wait(
                  timedelta(seconds=10)
              )
              print(
                  f"[global rank {rank}] {name} local rank = {dist.get_rank(group)}",
                  flush=True,
              )
      dist.barrier()

      for name, _, group in groups:
          if rank == 1:
              expected_rank = dist.get_rank(group)
              print(
                  f"[global rank 1] timing out {name}; "
                  f"expected WorkNCCL prefix rank = {expected_rank}",
                  flush=True,
              )
              try:
                  dist.all_reduce(tensor, group=group, async_op=True).wait(timeout)
              except torch.distributed.DistBackendError:
                  pass
          dist.barrier()

      dist.destroy_process_group()


  if __name__ == "__main__":
      main()
      ```