+++
title = "Why your terminal freezes on Lustre: Distributed storage, metadata storms, and recovery hangs"
date = '2026-09-01T17:23:07+08:00'
description = "An investigation into why interactive shells hang on HPC remote clusters due to Lustre metadata storms, OST failures, and uninterruptible I/O sleep."
summary = "A deep dive into how Lustre filesystem stalls freeze SSH terminals and interactive workflows, covering Lustre architecture (MDT/OST), recovery timeouts, common failure classes, and why small-file metadata operations trigger cluster-wide hangs."
draft = false
+++

This is a concrete, but annoying, problem I encountered when working on a remote cluster. I could still connect over SSH and the cursor blinked, but typing felt delayed, `cd` took seconds to respond, and `ls` appeared stuck. In one recent incident I investigated, the underlying issue was neither the terminal nor the OS, it was a stalled Lustre filesystem request.

This post explains how that happens, what I learned from the incident, and how to mitigate similar failures in distributed file systems.

## What you experience

The most visible symptom was that interactive work became nearly impossible. Standard Linux commands such as changing directories, listing files, and activating an environment which normally take no time became extremely slow or stopped responding entirely. When I ran a process dump, it showed Git operations were blocked in `cl_sync_io_wait`, a wait point in the Lustre client's kernel I/O path. This effectively blocked me from performing Git operations on my codebase (`git pull`, `git push`, etc.).

The process isn't necessarily CPU-bound, deadlocked in user code, or waiting for terminal input. Instead, it is sleeping inside the kernel while the Lustre client waits for a remote storage service or network path to recover.

This matters because interrupting the application doesn't solve the problem. Pressing `Control-C`, which normally terminates the process, didn't do anything. A process in Linux uninterruptible I/O sleep (usually shown as the D state) often cannot exit until the I/O path responds. Essentially, the frozen terminal is a symptom of an unresolved distributed-system failure, making it extremely frustrating for development and debugging.


## What is Lustre?

Lustre is an open-source, high-performance parallel filesystem. We use it in many of our internal clusters, sometimes through on-prem infrastructure and sometimes through vendor or cloud implementations like AWS or Azure.

The tool is designed for workloads such as distributed training, where many compute nodes need concurrent access to very large datasets and checkpoints. It achieves high throughput by spreading file data across many storage targets, so clients can transfer different chunks in parallel. This is great for large sequential data, but not very helpful for normal workloads that behave like a general-purpose home directory or workspace.

## Simple architectural model

Lustre separates **metadata** from **file data**:
- **Metadata Server (MDS)** answers questions about the namespace: what files and directories exist, who owns them, permissions, timestamps, directory entries, and where a file's data is located.
- A **Metadata Target (MDT)** is the persistent target that holds the metadata.
- An **Object Storage Server (OSS)** hosts and manages storage hardware.
- The **Object Storage Targets (OSTs)** are the data targets managed by OSSs.
- Each compute or login node runs a **Lustre client**. It communicates with the services over **LNet** (Lustre's network layer).

Before accessing data, the client checks the "catalog" to see where it is located. Operations like `ls` and `cd` normally require metadata work. Once a client knows the layout of a file, it reads or writes file data directly against one or more OSTs. Essentially, MDTs live on MDSs, and OSTs live on OSSs; the former hold the actual data/metadata, and the latter are the servers hosting them.

This leads to the following investigation steps:
1. If basic navigation (`cd`, `ls`) fails, suspect the metadata service, MDT, or metadata-path connectivity.
2. If only certain files or paths hang during reads or writes, suspect the OSTs holding those files or their network path.
3. If only one host is affected while peers are healthy, suspect that specific client, its NIC, LNet configuration, or the local node.

## Why can slow typing be caused by storage I/O?

The short answer is that the shell actually does much more filesystem work than it appears to:

- The current working directory must be resolved, meaning Lustre has to confirm where you are.
- A rich prompt may query Git state (which requires accessing files in `.git`).
- A remote editor (if connected via VS Code) continuously watches project trees, resolves symlinks, and reads configuration files.

If any of this state lives on a stalled Lustre mount, interaction becomes sluggish even when the keyboard, SSH transport, CPU, and RAM are all perfectly healthy. The command line feels frozen because the shell hasn't reached the point where it can draw the next prompt.

## Why does Lustre hang instead of failing fast?

Distributed filesystems must preserve consistency through failures. This means that if a target is briefly unavailable, immediately returning an error could turn a transient interruption into a failed job or an inconsistent operation. Lustre therefore pauses clients while a server, target, or failover partner recovers.

During this recovery, clients reconnect and replay eligible state and transactions. This is normally a feature, but it means application threads can wait for minutes. If a client cannot rejoin before the recovery window expires, it may be evicted. In-progress I/O can fail, and cached writes that were not durably stored can be lost. Recovery completion is designed to preserve filesystem consistency even when an individual client fails to participate. Typically, the recovery window is set to 5 minutes for soft recovery and 15 minutes for hard recovery.

For an OST failure specifically, the default behavior is to block I/O targeting that OST while the target recovers or fails over. For an MDT failure, namespace operations can block much more broadly. That is why an MDT-side incident can make a cluster feel globally frozen even if overall storage capacity and aggregate data throughput look reasonable.

### Common failure classes

#### A failed or unstable OST

If an OST becomes unavailable, files whose data uses that target can hang or fail while unrelated files continue to work. 

During a real incident this year, unresponsive Lustre OSTs caused jobs to hang, produced `Kill Task Failed` errors, and triggered cascading node drains. The incident was resolved by **rebooting a hung storage node and failing over an unstable OST to a healthy spare.** The root fix has to be storage-side; restarting a user process does not address the issue.

#### An MDT metadata stall or lock contention event 

These errors are tricky because almost any routine operation may touch metadata. During a real event on AWS, filesystem operations hung while the service investigation pointed to lock contention around an MDT responsible for quota-related activity. We had to restart affected metadata services to restore traffic.

#### A client-side connectivity failure

Sometimes the filesystem is healthy for everyone, but one node can't reach it reliably. In that case, the problem may be a bad NIC, packet errors, or LNet config issues.

## Takeaways

I think the key point is that Lustre is not optimized for huge numbers of small-file metadata operations.

For every file, operations such as `stat`, `open`, `create`, `rename`, `unlink`, and directory traversal require metadata RPCs (Remote Procedure Calls: one computer asks another over the network to perform an operation and return the result) and distributed locks. A single large 100GB object may be easy for Lustre to stream in parallel. A million 1KB files can be hard because the bottleneck is no longer transfer bandwidth; it becomes millions of lookups and lock operations (both compute and network bound).

A metadata storm occurs when many clients issue metadata calls at the same time. Typical examples include:
- `find /lustre` or another broad tree walk
- `ls -l` on a directory with a huge number of entries
- recursive `du`, `grep`, `rsync`, backup traversal, `chmod -R`, or `chown -R`
- IDE file indexing or agent workflows that repeatedly inspect large trees
- dependency installation, builds, or virtual environments placed on a shared filesystem

An unlink storm is the deletion counterpart: a large `rm -rf`, or many jobs cleaning up temporary files simultaneously. This can serialize on parent-directory locks and overload the MDT.

These storms increase latency, queue RPCs, delay lock callbacks, and can trigger client evictions or recovery behavior under pressure. This turns into a vicious feedback loop: an overloaded service delays clients, delayed clients accumulate work or time out, and the resulting recovery and reconnect activity adds even more work to the system.

### What to do when your terminal freezes

When an incident is active and your terminal is sluggish, the goal is to confirm whether the storage mount is actually frozen without running commands that will lock up your shell completely.

> **Crucial First Step:** Change directory out of the Lustre path immediately (e.g., `cd ~` or `cd /tmp`). Running diagnostic commands while sitting inside a stalled folder can instantly freeze your remaining active shell.

#### 1. Run safe, bounded diagnostics
Run these non-blocking commands from a local (non-Lustre) directory. Wrapping them in `timeout` ensures your terminal won't hang indefinitely if the storage isn't responding:

```bash
P=/lustre/fsw  # Replace with your affected mount path

date
hostname
timeout 15 stat "$P"; echo "stat_exit=$?"
timeout 15 ls -ld "$P"; echo "ls_exit=$?"
timeout 20 lfs df -h "$P"; echo "lfsdf_exit=$?"
timeout 20 lfs check servers; echo "check_servers_exit=$?"

# Check kernel logs for connection issues or timeouts
dmesg -T | grep -iE 'lustre|lnet|timeout|reconnect|evict|slow reply' | tail -200
```

An exit status of `124` means the command timed out. This is clear proof that the path is stalled, though you'll still need storage-side logs to determine whether it's a network, client, or backend storage failure.

#### 2. Gather context for your admin or storage team
If the mount is unresponsive, gather the following details before submitting a support ticket:

- **Environment:** Cluster name, exact hostname, and affected Lustre path.
- **Symptom:** Is it a metadata stall (`ls`, `cd`, `python import`) or a data read/write stall?
- **Timeline:** Start time (preferably in UTC) and whether it is continuous or intermittent.
- **Impact:** Affected Slurm/PBS job IDs and node lists.
- **Scope:** Does opening a new SSH session, VS Code window, or testing on a different node reproduce the problem?
- **Logs:** Relevant `dmesg` kernel errors and the stack trace of any hung process (`cat /proc/<PID>/stack`, if accessible).

*(Note: Always run client triage directly on the host machine rather than inside a container, as bind-mounted containers can mask or alter kernel-level storage errors.)*