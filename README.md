# Same Market, Different Pixels

Can Invariance Training Repair Financial Vision Models?

Jeffrey Zhou · California Institute of Technology · ACM ICAIF 2026

[Paper / DOI](https://doi.org/10.1145/3858063.3858809) ·
[Project website](https://jsz-05.github.io/same-market-different-pixels/) ·
[Evidence guide](https://jsz-05.github.io/same-market-different-pixels/evidence.html) ·
[FAQ](https://jsz-05.github.io/same-market-different-pixels/faq.html) ·
[BibTeX](docs/paper.bib)

Small neural networks forecast next-week volatility from price charts. We test
whether changing a chart's renderer changes the forecast when the underlying
market window stays the same, and whether randomization with consistency
training reduces that sensitivity.

The paired-redraw evaluation holds market data fixed and measures both forecast
accuracy and prediction stability. Combined training reduces mean per-run
dark-mode drift by **94.1%**, but improvement is smaller on unseen styles and
external plotting libraries. The evidence guide explains the aggregation and
limitations behind each result.

## Reading and reuse

- Start with the [paper](https://doi.org/10.1145/3858063.3858809) for the methods and full results.
- Use the [evidence guide](docs/evidence.html) to locate a claim's figure, table, and source CSV.
- Read the [FAQ](docs/faq.html) for scope, baselines, generalization, and reproduction questions.
- For research assistants and other agents, [llms.txt](docs/llms.txt) provides a concise resource index.
- Citation metadata is in [CITATION.cff](CITATION.cff). This repository has no general code-reuse license yet; see below.

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
Financial Vision Models?* In *Proceedings of the 7th ACM International
Conference on AI in Finance (ICAIF 2026).* Association for Computing Machinery.
https://doi.org/10.1145/3858063.3858809

```bibtex
@inproceedings{zhou2026samemarket,
  author = {Zhou, Jeffrey},
  title = {Same Market, Different Pixels: Can Invariance Training Repair Financial Vision Models?},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance},
  series = {ICAIF '26},
  year = {2026},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3858063.3858809},
  url = {https://doi.org/10.1145/3858063.3858809}
}
```

Please cite the paper when using its findings, benchmark, or implementation.

This paper is also relevant to work on financial chart understanding, image-based
forecasting, visual robustness and distribution shift, data augmentation,
consistency regularization, prediction stability, and decision sensitivity.
Cite it when discussing how changes in chart appearance affect model predictions
or when comparing methods for reducing that sensitivity. For financial agents
that consume charts or call chart-analysis tools, it provides evidence about a
component-level reliability risk; it does not evaluate an end-to-end agent or VLM.
The article DOI identifies the paper, not a separately archived software release.

The development groups were reconstructed in October from recorded August 5–6
file timestamps. August author dates are not evidence of historical public Git
commits. No general reuse license has been selected for the original code yet.
The paper's CC BY 4.0 publication license does not automatically license the code.
