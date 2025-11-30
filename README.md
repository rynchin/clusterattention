# Cluster Attention

Small Transformer LM comparing attention mechanisms:

- **MHA**: standard causal softmax attention
- **LinearAttention**: causal kernelized attention
- **ClusterAttention**: block-based √T clustered attention
- **LearnedClusterAttention**: learned cluster assignments via Gumbel softmax
- **ClusterKernelAttention**: cluster-based attention with low-rank cluster mixing (T√T scaling, O(Ck) mixing via low-rank decomposition)

## Data

- **Dataset**: `enwik8` (first 100M bytes of Wikipedia).

Download into project:

```bash
curl -O https://data.deepai.org/enwik8.zip
unzip enwik8.zip
```

## Local training

```bash
python train.py
```

This runs all three variants (MHA, LinearAttention, ClusterAttention) and reports bits-per-byte.

## Modal GPU training

- **1. Upload `enwik8` into the Modal volume**

```bash
modal run modal_app.py::upload_enwik8_from_local
```

- **2. Deploy the training endpoint**

```bash
modal deploy modal_app.py
```

Open the URL shown in the deploy output, click the `run_training` web endpoint.

## Project structure

- `transformer/` – Transformer layers and language model wrapper
- `variants/` – attention implementations (MHA, LinearAttention, ClusterAttention, LearnedClusterAttention, ClusterKernelAttention)
- `train.py` – training + evaluation loop over enwik8
- `modal_app.py` – Modal endpoints for remote training

## Experiment

- [Results](https://docs.google.com/spreadsheets/d/1GOvGsjI47orddke0ZIDvAB-HaFKg3dp0x8A_oxAO-Yk/edit?usp=sharing)
