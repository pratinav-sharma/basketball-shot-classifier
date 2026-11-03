# Profile after the baseline works

No NVIDIA GPU or Nsight executable was detected in the initial cloud workspace. CPU setup and correctness checks can proceed now. No paid hardware has been provisioned.

## When NVIDIA hardware is available

Use an existing authorized NVIDIA machine. Run `nvidia-smi`, install a PyTorch build compatible with its driver using the [official installation selector](https://pytorch.org/get-started/locally/), and run:

```bash
python -m shot_classifier doctor
nsys --version
```

The current environment's CPU-only PyTorch build cannot use CUDA even if hardware later becomes available. `doctor` must report `cuda_available: true` before GPU work. Nsight Systems must also be installed separately. Hardware, driver, profiling permissions, and toolkit compatibility need checking on that machine.

## Start with the timeline

Nsight Systems shows CPU activity, CUDA calls, GPU kernels, and transfers over time. It helps answer: is video decoding keeping the GPU idle, are copies expensive, or is the network computation the main cost? Nsight Compute examines individual GPU kernels in more detail; use it only when the timeline identifies a kernel worth studying.

Run an ordinary CUDA training pass first to populate the pretrained-weight cache and confirm correctness. Then collect a small representative run with a fresh output directory:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --output=runs/shot_profile python -m shot_classifier train data/manifest.csv --device cuda --epochs 5 --output runs/profile_run
nsys stats runs/shot_profile.nsys-rep
```

This is an initial whole-process trace, not a steady-state benchmark. It includes decoding, first-use initialization, feature extraction, training, and validation. This starter does not emit custom NVTX regions yet. Use several representative clips and inspect later operations after warm-up; for precise benchmarking, add named regions around decoding, transfers, encoding, and head updates, then capture repeated warmed iterations. See NVIDIA's [focused profiling and NVTX guidance](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

The frozen-feature baseline decodes/encodes each train and validation clip once, then runs many tiny head updates. Those phases have different bottlenecks. More GPU activity does not automatically mean a better end-to-end system. Record clips/second, total runtime, peak memory, hardware, versions, clip lengths, and batch size. GPU operations are asynchronous: use CUDA events or synchronize around wall-clock timing.

First try an evidence-supported change such as batching encodes, reducing repeated decoding, or adjusting data loading. Recheck predictions and validation metrics after any change. Only consider mixed precision, compilation, or custom CUDA kernels after profiling shows where time is spent and simpler changes have been measured. No CUDA optimization or GPU performance claim has been made for this starter.
