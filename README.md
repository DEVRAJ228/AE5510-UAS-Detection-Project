# AE5510 – Security of Safety-Critical Systems
## UAS object detection and printable adversarial-patch study



## Structure
```text
AE5510-UAS-Detection-Project/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── model_training.ipynb
│   ├── patch_training.ipynb
│   └── evaluation.ipynb
├── src/
│   ├── dataset_audit.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   └── sample_metrics.csv
└── references/
    ├── uas-ab-original.ipynb
    └── uas-yolo-original.ipynb
```

## Environment and dependencies
The source notebooks were written for Kaggle and use `/kaggle/input` and `/kaggle/working`. Patch optimisation and detector training require a CUDA-capable GPU. Install packages with `pip install -r requirements.txt`. For strict reproduction, record and pin the exact versions reported by the runtime used for the experiment.

## Suggested workflow
1. Run `notebooks/data_preparation.ipynb` with the aerial dataset attached.
2. Use `notebooks/model_training.ipynb` for the selected YOLO training experiment. Some cells define alternative configurations and cross-validation; they are not all meant to run sequentially.
3. Run `notebooks/patch_training.ipynb` for the printable patch workflow.
4. Run the relevant experiment group in `notebooks/evaluation.ipynb` for transfer evaluation, patch examples, robustness and phone-camera analysis.

Read notebook introductions and source-cell headings before running. Later cells depend on variables, paths and checkpoints produced earlier. These are curated source notebooks, not standalone one-click scripts.

## External inputs not included
This ZIP does not include the aerial dataset, trained detector weights, patch checkpoint `.pt` files or personal phone photographs. Attach authorised inputs separately in Kaggle. Evaluation cells require the corresponding weights/checkpoints; without them, those cells cannot execute.

## Results
`results/sample_metrics.csv` is a header-only template. No unverified or invented measurements are included.

## Source attribution and public release
The curated code is extracted from the supplied Kaggle notebooks. Before making this repository public, confirm that you have permission to redistribute the notebook code and comply with its Kaggle licence and attribution terms. Do not upload credentials, restricted datasets or weights without permission.

AE5510 – Security of Safety-Critical Systems, IIT Madras.
