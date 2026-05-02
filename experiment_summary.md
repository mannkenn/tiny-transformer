## Experiment Summary

| Run Name | Change | Batch | Grad Accum | Eff Batch | Precision | Compile | Tokens/sec | Step Time (s) | Max Mem (GB) | Val Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke Test | debug test on cpu | 4 | 1 | 4 | fp32 | no | 14k | 0.008 | 0.02 | 2.9837 |
| Baseline | baseline hyperparams | 64 | 1 | 64 | fp32 | no | 291k | 0.0569 | 3.6 | 1.6111 |
| Decrease to batch size 32 | batch size 32 | 32 | 1 | 32 | fp32 | no | 227k | 0.0377 | 1.898 | 1.5159 |
| Increase to batch size 128 | batch size 128 | 128 | 1 | 128 | fp32 | no | 275k | 0.1194 | 7.048 | 1.8073 |
| Increase to batch size 256 | batch size 256 | 256 | 1 | 256 | fp32 | no | 240k | 0.2732 | 13.942 | 2.0223 |
| Gradient accumulation over 4 steps | grad accum | 16 | 4 | 64 | fp32 | no | 124k | 0.1363 | 1.068 | 1.6227 |
| Gradient accumulation over 2 steps | grad accum | 32 | 2 | 64 | fp32 | no | 245k | 0.0678 | 1.944 | 1.6044 |
| BF16 Mixed Precision| bf16 mp | 64 | 1 | 64 | bf16 | no | 291k | 0.0558 | 3.6 | 1.6111 |
| Flash Attention | 64 | 1 | 64 | fp32 | no | 293k | 0.057 | 3.597 | 1.6112 |
| torch.compile | 64 | 1 | 64 | fp32 | yes | 291k | 0.0551 | 3.6 | 1.6111 |
