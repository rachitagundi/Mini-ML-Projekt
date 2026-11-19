# Mini ML Projekt: Healthcare predictions explained

A small Python project investigating whether a model can estimate a diabetes-progression target more accurately than a simple average. The results are presented in a visual, plain-language HTML report.

This project develops healthcare data analysis skills. EnergyAtlas remains a separate energy project.

## Question and data

Using ten numerical inputs from scikit-learn's bundled diabetes dataset, predict a quantitative measure of disease progression one year after baseline.

This is regression: predicting a number. It is not diabetes diagnosis, cancer detection, or a probability of developing disease. The project is an educational analysis and has not been clinically validated.

Dataset reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

## Run on Windows

In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
Start-Process .\outputs\report.html
```

Skip environment creation if .venv already exists. Python 3.12 is the intended starting environment. No GPU, API key, or separate dataset download is required.

## Visual results

The self-contained report opens locally in a browser and includes:

- Record counts for training and testing.
- A bar chart comparing average prediction errors.
- Recorded outcomes and predictions for twelve test examples.
- Plain-language explanations of the method and limitations.

The report contains embedded chart images and requires no external scripts or chart services.

## Method

1. Hold out 20% of records before training, using random_state=42.
2. Train a baseline that predicts the training target average.
3. Train Ridge regression with alpha=10 and feature scaling inside a pipeline.
4. Compare both approaches on the same held-out records.
5. Save scores, prediction errors, plots, and the visual report.

The scaler is fitted only on training data. The test split is for evaluation, not repeated model selection.

## Verified example results

A run with scikit-learn 1.8.0 produced:

| Approach | Mean absolute error | R² |
|---|---:|---:|
| Training-average baseline | 64.01 | -0.012 |
| Ridge regression | 42.86 | 0.457 |

The model had approximately 33% lower average absolute error than the baseline on this split. This is not 33% diagnostic accuracy. Errors are in the dataset's target-score units.

One split does not establish performance in another population or clinical usefulness. Large individual errors can remain despite better average performance. These data do not establish causes of disease progression.

## Files

| File | Purpose |
|---|---|
| train.py | Train and evaluate the baseline and model |
| report.py | Build charts and the HTML results page |
| requirements.txt | Package requirements |
| learning-notes.md | First-exercise concepts and questions |
| .gitignore | Exclude environments and generated outputs |

Generated files in outputs/ include report.html, comparison.png, examples.png, predictions.png, metrics.json, and prediction_errors.csv.

## Next learning steps

- Run the project locally and inspect the charts.
- Record personal observations in learning-notes.md.
- Use cross-validation within training data to investigate stability.
- Inspect errors across target ranges.
- Compare a second model using training-only validation.
- Document conclusions without presenting the model as a clinical tool.

No repository licence has been selected.
