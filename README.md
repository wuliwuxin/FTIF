# Taming Multi-Dimensional Tail: A Factorized Tail-Interaction Framework for Hyperscale Cloud Workload Forecasting



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

## Citation

If you find this work useful, please consider citing our paper:
```
@inproceedings{2026WuFTIF,
  title={Taming Multi-Dimensional Tail: A Factorized Tail-Interaction Framework for Hyperscale Cloud Workload Forecasting},
  author={Wu, Xin and Teng, Fei and Yang, Chen and Li, Tianrui},
  booktitle={Proceedings of the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD‘26)},
  year={2026}
}
```