"""Create a self-contained, plain-language HTML results page."""
import base64
from pathlib import Path
import matplotlib.pyplot as plt

def write_report(output, metrics, errors, train_count, test_count):
    baseline = metrics["baseline"]["mae"]
    model = metrics["ridge"]["mae"]
    improvement = 100 * (baseline - model) / baseline
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(["Always predict the average", "Learn from the input data"],
                   [baseline, model], color=["#94a3b8", "#0d9488"])
    ax.bar_label(bars, fmt="%.2f", padding=6)
    ax.set(xlabel="Average prediction error in target-score units (lower is better)",
           xlim=(0, baseline * 1.2))
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    sample = errors.sort_index().head(12)
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = list(range(len(sample)))
    ax.plot(positions, sample["actual"], "o-", label="Recorded outcome", color="#334155")
    ax.plot(positions, sample["predicted"], "o--", label="Model prediction", color="#0d9488")
    ax.set(xticks=positions, xticklabels=[str(i+1) for i in positions],
           xlabel="Example test records (first 12 by dataset index)",
           ylabel="Disease-progression target score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "examples.png", dpi=160)
    plt.close(fig)
    def embed(name):
        encoded = base64.b64encode((output / name).read_bytes()).decode("ascii")
        return "data:image/png;base64," + encoded
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Healthcare ML | Results explained</title>
<style>
body{{margin:0;background:#f1f5f9;color:#183044;font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:40px 22px}}
header{{background:#123c48;color:white;padding:36px;border-radius:20px}}
h1{{font-size:clamp(28px,5vw,44px);line-height:1.15;margin:12px 0}}
.tag{{color:#99f6e4;font-size:14px;letter-spacing:1px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin:24px 0}}
.card,section{{background:white;padding:26px;border-radius:16px;margin-bottom:20px}}
.card{{margin:0}} .number{{font-size:36px;font-weight:700;color:#0f766e}}
h2{{font-size:24px;line-height:1.3}}img{{width:100%;height:auto}}
.note{{border-left:5px solid #d97706;padding-left:18px}}
small{{color:#475569}} a{{color:#0f766e}}
</style></head><body><main>
<header><div class="tag">MINI ML PROJEKT · HEALTHCARE DATA</div>
<h1>Can a model make better predictions than an average?</h1>
<p>A learning project using ten numerical inputs to estimate a recorded
diabetes-progression score one year after baseline.</p></header>
<div class="grid">
<div class="card"><div class="number">{train_count}</div>records used for learning</div>
<div class="card"><div class="number">{test_count}</div>separate records used for testing</div>
<div class="card"><div class="number">{improvement:.1f}%</div>lower average error than the baseline on this test split</div>
</div>
<section><h2>1. What problem did we investigate?</h2>
<p>We compared two approaches on the same held-out records. The baseline gives
everyone the training average. Ridge regression learns relationships between
the inputs and the recorded outcome.</p>
<p>The outcome is a numerical research target. It is not a probability of getting
diabetes, a diagnosis, or a percentage of disease severity.</p></section>
<section><h2>2. Did learning from data help?</h2>
<p>The baseline was off by <strong>{baseline:.2f} target-score units</strong> on average.
The model was off by <strong>{model:.2f} units</strong>.
A shorter bar means smaller mistakes.</p>
<img src="{embed('comparison.png')}" alt="Comparison of average prediction errors for baseline and Ridge regression">
<p><strong>{improvement:.1f}% lower error does not mean {improvement:.1f}% diagnostic accuracy.</strong>
It is the relative reduction in mean absolute error compared with this baseline.</p></section>
<section><h2>3. What do individual predictions look like?</h2>
<p>Each position below is a different held-out record, not a time step.
A gap between the two lines is a prediction error. These are the first twelve
test records by dataset index, selected without ranking their prediction quality.</p>
<img src="{embed('examples.png')}" alt="Recorded outcomes and model predictions for twelve held-out records">
<p>The largest absolute error across all test records was
<strong>{errors['absolute_error'].max():.2f} units</strong>.
Average performance can hide large individual mistakes.</p></section>
<section><h2>4. How was the comparison kept fair?</h2>
<p>We held back 20% of the records before training. The model and its scaler
learned only from the other 80%. Both approaches were scored on exactly the same
test records. The split uses a fixed seed so the exercise can be repeated.</p></section>
<section><h2>5. What can we conclude?</h2>
<p>The model outperformed this simple baseline on one test split.
This does not establish clinical usefulness or performance on a new population.
The dataset alone does not tell us why an individual prediction was wrong.</p>
<p class="note">Educational analysis only. No clinical validation, diagnosis,
treatment recommendation, or cancer prediction is provided.</p>
<p>Next: use cross-validation within the training data, inspect errors across
target ranges, and document whether the result is stable.</p></section>
<footer><small>Source:
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html">scikit-learn diabetes dataset</a>.
Generated from the local run. No external scripts or chart services are required.</small></footer>
</main></body></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
