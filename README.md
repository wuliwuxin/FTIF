# FTIF: Factorized Tail-Interaction Framework

Multi-dimensional tail prediction for cloud systems.

## Install

```bash
pip install -r requirements.txt
```

## Data

- `datasets/GenTD26/`
- `datasets/Spot26/`

## Run

```bash
python main.py --mode GenTD26 --device cuda
python main.py --mode Spot26 --device cuda
python main.py --mode both --device cuda
```

Use `--target`, `--epochs`, `--batch_size`, `--lr`, `--seed` as needed.

## Output

- **results/** — `<exp_name>/<target>/metrics.csv`, `summary_report.txt`, prediction npz; `consolidated_results.csv` for all runs.
- **checkpoints/** — best model `FTIF_<dataset>.pth`.

