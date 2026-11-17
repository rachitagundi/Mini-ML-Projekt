"""A small regression exercise. Run with: python train.py"""
from pathlib import Path
import json
from report import write_report
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    # Bundled educational data: no download or personal data upload required.
    data = load_diabetes(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }
    output = Path(__file__).resolve().parent / "outputs"
    output.mkdir(exist_ok=True)
    metrics = {}
    for name, model in models.items():
        # Fit only on training data, including the scaler inside the pipeline.
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
        print(f"{name}: MAE={metrics[name]['mae']:.2f}, R2={metrics[name]['r2']:.3f}")
    prediction = models["ridge"].predict(X_test)
    errors = pd.DataFrame({
        "actual": y_test, "predicted": prediction,
        "absolute_error": abs(y_test - prediction),
    }).sort_values("absolute_error", ascending=False)
    errors.to_csv(output / "prediction_errors.csv", index_label="sample_index")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots()
    ax.scatter(y_test, prediction, alpha=0.7)
    limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    ax.plot(limits, limits, "--", color="gray")
    ax.set(xlabel="Actual target", ylabel="Predicted target", title="Held-out predictions")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=150)
    plt.close(fig)
    write_report(output, metrics, errors, len(X_train), len(X_test))
    print(f"Open {output / 'report.html'} in your browser")
    print(f"Saved results to {output}")

if __name__ == "__main__":
    main()

