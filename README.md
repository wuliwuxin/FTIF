# Official implementation of "Taming Multi-Dimensional Tail: A Factorized Tail-Interaction Framework for Hyperscale Cloud Workload Forecasting"

FTIF is a PyTorch framework for tail-aware cloud workload forecasting. It targets multi-dimensional heavy-tail signals in hyperscale cloud systems and includes experiments on:

- **GenTD26**: generative AI serving traces, currently using GPU memory usage as the default target.
- **Spot26**: spot GPU workload traces, currently using job duration as the default target.

## Repository Structure

```text
FTIF/
├── main.py                 # Command-line entry point
├── config.py               # Dataset, model, training, and evaluation configs
├── model.py                # FTIF model and tail-focal loss
├── data_processing.py      # GenTD26 and Spot26 data processors
├── train.py                # Training, validation, checkpointing, and testing
├── evaluate.py             # Metrics, reports, and visualization helpers
├── metrics.py              # Metric implementations
├── utils.py                # Reproducibility, device, logging, and checkpoint utilities
├── requirements.txt        # Python dependencies
├── Manuscript.pdf          # Paper manuscript
├── datasets/
│   ├── GenTD26/            # GenAI serving trace data
│   └── Spot26/             # Spot GPU workload trace data
└── results/                # Example output plots and generated experiment results
```

## Environment

The code is tested with Python 3.8+ and PyTorch 2.0+.

Create an environment and install dependencies:

```bash
cd FTIF
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU experiments, install a PyTorch build that matches your CUDA version before installing the rest of the requirements. CPU and Apple Silicon MPS execution are also supported through `--device cpu` and `--device mps`.


## Quick Start

Run the GenTD26 experiment:

```bash
python main.py --mode GenTD26 --device cuda
```

Run the Spot26 experiment:

```bash
python main.py --mode Spot26 --device cuda
```

Run both datasets:

```bash
python main.py --mode both --device cuda
```

## Model Components

FTIF combines several modules for tail-aware forecasting:

- **Adaptive Trend Decoupler**: separates normal trend components from tail-enriched residual signals.
- **Factorized Resonance Module**: uses CP-style factorized high-order interactions to model correlated tail risks.
- **Tail-Sensitive Embedding**: enriches temporal representations with tail-aware information.
- **Tail-Focal Loss**: emphasizes high-quantile regions during optimization.

These components can be enabled or disabled through `ModelConfig`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{2026WuFTIF,
  title = {Taming Multi-Dimensional Tail: A Factorized Tail-Interaction Framework for Hyperscale Cloud Workload Forecasting},
  author = {Wu, Xin and Teng, Fei and Yang, Chen and Li, Tianrui},
  booktitle = {Proceedings of the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  year = {2026}
}
```