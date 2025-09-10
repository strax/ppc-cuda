# CUDA solutions for the Programming Parallel Computers course

This repository contains selected solutions for Aalto University's [Programming Parallel Computers](https://ppc.cs.aalto.fi/) course. Both of these solutions have the first place on the [performance contest leaderboard](https://ppc-exercises.cs.aalto.fi/course/open2025/contest).

## Correlated pairs

The `cp5` exercise involves computing the Pearson correlation coefficient between each row vector of the input matrix $\mathbf{X}$ in 32-bit precision.
The approach taken here is to first to compute the standardized matrix $\tilde{\mathbf{X}}$ and then compute its Gram matrix $\mathbf{K} = \tilde{\mathbf{X}}  \tilde{\mathbf{X}}^\top$.

We normalize each row of the input matrix by first computing their means and sums of squared deviations with a warp-collective single-precision Welford's algorithm, merge the warp statistics with XOR shuffles, and then do a second pass over the rows to normalize. The normalization kernel is grid-strided and launched as a single wave to reduce wave quantization effects.

For the exercise, it is only necessary to compute the upper triangular part of $\mathbf{K}$. The MMA kernel uses FMA instructions (due to the absence of F32 Tensor Cores on Turing) with software pipelining (shared memory and registers). The input matrix is first transposed to facilitate coalesced global loads. The kernel is fully shared memory bank conflict free and achieves ~92.5% compute throughput, or ~94% of cuBLAS for the same task.

## Radix sort

For the `so6` exercise, the task is to implement an efficient parallel algorithm for sorting 64-bit unsigned integers. We implement the Onesweep algorithm by Adinets and Merrill [1], which combines LSD radix sort with efficient inter-block communication. Our implementation achieves 5.67GB/s or 711.8Melem/s on-device throughput on the target hardware (NVIDIA Quadro RTX 4000).

[1]: Adinets, A., & Merrill, D. (2022). Onesweep: A faster least significant digit radix sort for gpus. [arXiv:2206.01784](https://arxiv.org/abs/2206.01784).