# SpatiaLQA
[CVPR 2026] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models

## Requirements
- Python 3.11
- Pytorch 2.7.0
- transformers 4.52.4

## Data
The dataset can be downloaded from https://huggingface.co/datasets/xyc99/SpatiaLQA.

## Example Workflow
### Step 1: Generate answers
```
python answer.py
```

### Step 2: Generate matching matrix
```
python gen_sim_matrix.py
```

### Step 3: Compute metrics
```
python metric.py
```
