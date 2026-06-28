# ResNet Structured Pruning Experiment

## Goal

This experiment aims to visually and empirically demonstrate SOTA structured physical pruning practices using a ResNet-18 model on a CIFAR-5 classification task.

The central hypothesis tested is:
1. **Redundancy Isolation:** CNN architectures like ResNet-18 contain massive channel redundancy, allowing substantial model compression without significant accuracy degradation.
2. **Layer Sensitivity:** Channels are not uniformly important; bottleneck layers (like transition convs) are highly sensitive to pruning, whereas residual-coupled layers can be pruned aggressively.
3. **Iterative Recovery:** Multi-step progressive pruning followed by recovery fine-tuning outperforms one-shot pruning by allowing the network's remaining parameters to adapt dynamically.

## Experiment details

### Model

We adapt a standard ResNet-18 model (originally designed for ImageNet) to work with $32 \times 32$ CIFAR images by modifying the input resolution bottleneck (using a $3 \times 3$ convolution with stride 1, padding 1, and bypassing the first max pooling layer).

The model is pruned using **structured channel pruning** (rather than weight masking), physically slicing the parameters to yield a smaller, faster model in memory:
$$\text{Weight Shape} = [\text{Out Channels}, \text{In Channels}, K_h, K_w] \longrightarrow [\text{Out Channels}', \text{In Channels}', K_h, K_w]$$

### Data

We train and evaluate on a custom **CIFAR-5** dataset containing the 5 animal classes: `['bird', 'cat', 'deer', 'dog', 'frog']` from the CIFAR-10 dataset, mapped to target labels `0-4`.

### Pruning Methods Used

1. **Layer Sensitivity Analysis:** Pruning one isolated layer at a time to trace its influence on validation accuracy and identify structural bottlenecks.
2. **First-Order Magnitude Importance (L1-norm):** Pruning channels with the smallest absolute weight sums.
3. **Iterative Pruning & Fine-Tuning:** Compressing the model in $5$ sequential steps, running $1$ epoch of fine-tuning at each step to stabilize accuracy.

---

## Results

### Overall Comparison

| Metric | Baseline | Pruned Model (50%) | Change (%) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 86.94% | 83.42% | -3.52% |
| **Parameters** | 11.23M | 2.74M | -75.5% |
| **FLOPs / MACs** | 550.0M | 128.9M | -76.6% |
| **Disk Size** | ~44.9 MB | ~11.0 MB | -75.5% |
| **Batch Latency (GPU)** | 13.92 ms | 5.11 ms | -63.3% |

### Key Observations

1. **Pruning as a Regularizer:** In the first iterative step (pruning ~13% of channels), the validation accuracy of the fine-tuned model actually **increased** from $86.94\%$ to **$87.42\%$**, proving that structured pruning can act as a regularizer by eliminating noisy, over-fitted channels.
2. **Channel Sensitivity Bottlenecks:** The sensitivity analysis identified **`layer2.0.conv1`** as the absolute most sensitive layer (dropping accuracy by **$17.08\%$** under isolated 40% pruning), whereas all `conv2` layers (coupled with block skip connections) survived with **$0.00\%$** drop. SOTA policies must freeze or lightly prune early transition convs.
3. **FLOPs-to-Speed Scaling (Quadratic Effect):** Structured channel pruning has a quadratic scaling effect on computational complexity and speed. By pruning $50\%$ of channels globally, both input and output dimensions of the conv matrices are halved, reducing overall FLOPs/MACs by **$76.6\%$** (nearly a $4\times$ reduction). Consequently, despite minor layer-wise GPU alignment overheads, the fully pruned model achieved a massive **$-63.3\%$ batch latency reduction** and a **$+172.6\%$ throughput increase** (2.7x speedup) on CUDA.
4. **Compression Efficiency:** We successfully compressed the parameter footprint by **$4\times$** while maintaining **$96\%$** of the baseline accuracy.
