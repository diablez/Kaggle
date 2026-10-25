# ENGE707 Facial Feature Classification

Reproducible PyTorch baseline for the 2026 Semester 2 Kaggle challenge. The model is an ImageNet-pretrained ResNet18 with a new eight-class output layer and 224 x 224 inputs.

## Data layout

Competition data is intentionally excluded from Git. After extracting the Kaggle archive, the expected local structure is:

```text
data/
  train_labels.csv
  sample_submission.csv
  train/train/*.jpg
  test/test/*.jpg
```

The supplied archive uses `train_labels.csv`, despite some assessment notes referring to `train.csv`. Test images are used only for final inference and never enter the train/validation split.

## Reproduce the environment (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name enge707-kaggle --display-name "Python (.venv ENGE707 Kaggle)"
```

`requirements-lock.txt` records the exact environment used for the evidence run. VS Code is configured to recommend the project-local interpreter. The code selects CUDA when available and otherwise runs on CPU; the initial machine check found a CPU-only PyTorch installation.

## Verify data and components

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py
.\.venv\Scripts\python.exe -m pytest -q
```

The audit validates CSV/image correspondence, opens every image, reports dimensions and formats, detects corrupt and duplicate-content files, records all eight exact labels, and writes a representative-class grid under ignored `artifacts/audit/`.

## Train the baseline

```powershell
.\.venv\Scripts\python.exe scripts\train.py --epochs 10 --batch-size 32
```

The split is stratified 80/20 with seed 42. Training uses random resized crop, horizontal flip, light colour jitter, ImageNet normalization, cross-entropy loss, AdamW, validation-loss scheduling, early stopping, and best-checkpoint selection. The validation path uses deterministic resize and centre crop. On a CPU-only machine, a faster first baseline can train only the new classifier head:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --freeze-backbone --epochs 5 --batch-size 32
```

Outputs include the exact split rows, configuration and package versions, epoch history, best checkpoint, classification metrics, confusion matrix, and learning curves. These live under ignored `artifacts/` to prevent large/generated files entering Git.

## Generate a submission

```powershell
.\.venv\Scripts\python.exe scripts\predict.py
```

Inference reloads the best checkpoint and follows `sample_submission.csv` row order exactly. Before writing, it checks column order, row count, image IDs, and that every prediction belongs to the checkpoint's eight-class vocabulary. Review the generated file before manually submitting it to Kaggle.

## Reproducibility notes

- Random seeds are fixed for Python, NumPy, PyTorch, and the stratified split.
- The train and validation row lists are saved with each run.
- Deterministic cuDNN behaviour is requested when CUDA exists.
- Pretrained ImageNet weights are permitted by the brief and downloaded from PyTorch on first use.
- Real run artifacts must be retained with their configuration; never copy metrics between runs.
- Do not commit `data/`, credentials, model weights, `artifacts/`, or submissions.

See `docs/report_evidence_outline.md` for the assessment evidence checklist.
