# Same Market, Different Pixels

Can Invariance Training Repair Financial Vision Models?

Jeffrey Zhou · California Institute of Technology · Accepted at ICAIF 2026

[Project website](https://jsz-05.github.io/same-market-different-pixels/)

Small neural networks forecast next-week volatility from price charts. We test
whether changing a chart's renderer changes the forecast when the underlying
market window stays the same, and whether randomization with consistency
training reduces that sensitivity.

## Contents

- `src/`: chart rendering, downloads, training, and analysis code.
- `tests/`: deterministic rendering, input construction, and normalization tests.
- `protocol/`: study specifications and data-retrieval manifests.
- `results/`: summary CSVs for the main ETF panel, external renderers, and stock replication.
- `docs/`: a small static GitHub Pages website.

This is a lightweight code-and-summary repository. Raw data, model weights,
row-level predictions, manuscript sources, submission files, and release
archives are not included. The scientific code and summary CSVs are unchanged
from the accepted study.

## Selected findings

These comparisons use three-seed ensembles. Drift is the mean absolute change
in predicted log volatility between paired chart views.

| Comparison | Canonical training | Randomization + consistency |
| --- | ---: | ---: |
| Custom dark-view drift | 0.705 | 0.029 |
| Custom dark-view alert flips | 42.8% | 1.8% |
| External-renderer worst drift | 0.543 | 0.347 |
| Alert flips at each method's drift-worst external renderer | 15.3% | 19.8% |
| Stock-panel dark-view drift | 0.466 | 0.017 |

External generalization is partial. The worst renderer is selected separately
for each method by drift, and lower drift does not guarantee fewer alert flips.
Simple preprocessing also helps; the numerical baseline remains invariant to
rendering and has slightly lower forecasting error than the image models.

The main test contains 1,036 windows across 37 ETFs and 28 dates in
January–July 2026. Development uses 2022–2025. This is a historical test, not a
prospective trading experiment or independently verified public preregistration.

## Run

Use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
make test
```

To regenerate local data and run outputs:

```bash
make download
make study
make analysis
make external-validation
make stock-replication
make extension-analysis
```

These commands download data and train models; they are not needed to read the
included summary CSVs. Generated weights and predictions remain local and are
ignored by Git. Training uses Apple MPS when available and otherwise CPU. No
paid API or external GPU is required. External rendering requires Chrome for
Kaleido. New Yahoo downloads or current stock constituents may differ from the
preserved retrieval manifests.

The original candle-body-width randomization collapses after pixel quantization;
that disclosed limitation is preserved in the code.

## Citation and history

Jeffrey Zhou. 2026. *Same Market, Different Pixels: Can Invariance Training Repair
Financial Vision Models?* Accepted at the 7th ACM International Conference on
AI in Finance (ICAIF 2026).

The development groups were reconstructed in October from recorded August 5–6
file timestamps. August author dates are not evidence of historical public Git
commits. No general reuse license has been selected for the original code yet.
