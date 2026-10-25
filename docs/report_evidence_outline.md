# ENGE707 report evidence inventory and outline

Target length: **3,000–3,500 words**. This file inventories reproducible evidence; the student's independent interpretation, justification, and reflection must be written by the student. The precise task headings should be reconciled with the official marking sheet before finalising the report because that document is not present in the repository.

## Task 1 — Problem and data understanding

- Competition objective, eight exact class names, and permitted data scope.
- `artifacts/audit/dataset_report.json`: row/file counts, formats, dimensions, missing/corrupt/duplicate checks, and class distribution.
- `artifacts/audit/representative_classes.png`: one labelled example per class.
- Discuss what the observed balance and image properties imply; do not infer hidden test labels.

## Task 2 — Data preparation

- `src/enge707/data.py`: ID-to-file mapping, RGB conversion, 224 x 224 transforms, and ImageNet normalization.
- `artifacts/baseline/train_split.csv` and `validation_split.csv`: exact stratified 80/20 split, seed 42.
- Explain why augmentation is training-only and why the unlabelled test set remains isolated.

## Task 3 — Baseline model design

- `src/enge707/model.py`: pretrained ResNet18 and replacement eight-output classifier.
- `artifacts/baseline/config.json`: actual model, optimizer, loss, scheduler, seed, package versions, device, and sample counts.
- Explain transfer learning, cross-entropy, AdamW, and CPU/CUDA fallback in your own words.

## Task 4 — Training and experiment procedure

- `scripts/train.py`: epoch loop, validation, checkpoint selection, learning-rate reduction, and early stopping.
- `artifacts/baseline/history.csv` and `learning_curves.png`: actual epoch evidence.
- Distinguish smoke tests from the full baseline and report only configurations that actually ran.

## Task 5 — Evaluation and analysis

- `artifacts/baseline/metrics.json`: validation loss, accuracy, per-class precision/recall/F1, macro and weighted averages.
- `artifacts/baseline/confusion_matrix.png`: actual confusion patterns.
- Identify weak classes from the recorded results and discuss plausible, evidence-linked causes without claiming causation.

## Task 6 — Test inference and Kaggle submission

- `scripts/predict.py`: best-checkpoint reload and deterministic inference.
- `artifacts/baseline/submission.csv`: checked row count, columns, ID order, and class vocabulary.
- Record the submission timestamp and Kaggle score only after the student submits it; do not invent a score.

## Task 7 — Reproducibility, limitations, and reflection

- `requirements-lock.txt`, `.vscode/settings.json`, README commands, Git branch/commits, and tests.
- Hardware limitations, runtime, possible data leakage controls, remaining uncertainty, and justified next experiments.
- Student-authored reflection: what worked, what did not, ethical/academic-integrity choices, and how evidence changed decisions.

## Final evidence checks

- Every reported number traces to a named artifact and run configuration.
- Figures have readable labels/captions and are referenced in the prose.
- Smoke-test metrics are never presented as final baseline performance.
- Kaggle score is included only if actually observed after manual submission.
- Word count is within 3,000–3,500 words and the official seven task names/order are used.
