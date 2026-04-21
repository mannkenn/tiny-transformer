# tiny-transformer

this project explores how to train and optimize transformer models under hardware constraints, with a focus on maximizing throughput and efficiency on a single GPU.

rather than scaling out across multiple devices, this work investigates how far we can push performance on limited compute, treating efficiency as a first-class objective alongside model quality.

## goals
- build a clean, minimal transformer training pipeline from scratch (PyTorch)
- optimize training performance on a single GPU (RTX 4090)

### identify and remove bottlenecks in:
- data loading
- memory usage
- compute efficiency
- measure and improve tokens/sec, GPU utilization, and training stability
