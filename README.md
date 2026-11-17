# Mini ML Projekt

A small Python project for learning regression: train a model, compare it with a simple baseline, and inspect its mistakes.

## What it does

The exercise uses scikit-learn's bundled diabetes dataset. Ten numerical features are used to predict a quantitative measure of disease progression one year after baseline. This is an educational example, not a medical decision tool or an energy forecasting model.

- Splits data into 80% training and 20% test data with a fixed random seed.
- Compares a training-mean baseline with Ridge regression.
- Fits feature scaling only on training data using a pipeline.
- Saves evaluation metrics, prediction errors, and a plot.

## Run on Windows

Python 3.12 is the intended starting environment. No GPU or API key is needed.

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

If the virtual environment already exists, skip its creation.

## Initial verification

A run with scikit-learn 1.8.0 produced:

| Model | Mean absolute error (lower is better) | R² (higher is better) |
|---|---:|---:|
| Training-mean baseline | 64.01 | -0.012 |
| Ridge regression | 42.86 | 0.457 |

MAE is measured in the dataset's target units, not a percentage. These results describe one fixed test split; they do not establish performance on other populations. The test data should not be repeatedly used to choose model settings.

## Generated outputs

After running the script, open the local `outputs` folder:

- `metrics.json`: evaluation scores.
- `prediction_errors.csv`: predictions sorted by largest absolute error.
- `predictions.png`: actual versus predicted values; points near the diagonal are more accurate.

Generated outputs and the virtual environment are excluded from Git.

## Project files

| File | Purpose |
|---|---|
| train.py | Load data, train models, evaluate, and save outputs |
| requirements.txt | Required Python packages |
| learning-notes.md | Concepts and questions for the first exercise |
| .gitignore | Keep local environments and generated files out of commits |

## Next steps

Run locally and explain the largest prediction errors. Then practise cross-validation on training data, compare a second model, and apply the process to a suitable public energy dataset. Agent development will be a separate step using existing language models and tools.

Starter implementation and documentation were prepared with AI assistance. Running the example is the beginning of the learning exercise.

## Dataset reference

[scikit-learn diabetes dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

No repository licence has been selected yet.
