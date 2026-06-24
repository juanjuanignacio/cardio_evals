# Mechanistic Interpretability of the Self-Preference Bias

This subpackage reproduces **Figures 3c–d** and **Figure 4** of the paper, along with the
activation-patching analysis used in Methods to localise the bias to layer 15 of
`Llama3.1:8b`.

Everything is self-contained: it has its own `src/` helpers, its own YAML config, and
reads only one secret (`HF_TOKEN`) from the project-level `.env`.

## What is reproduced

| Paper element | Script | Output |
|---|---|---|
| **Figure 3c** — within-tertile shuffle | `06_shuffle_within_tertile.py` | `results/tables/{model}_shuffled_within_tertile.json` |
| **Figure 3d** — cross-tertile shuffle (Δ score) | `07_cross_tertile_shuffle.py` | `results/tables/{model}_cross_tertile_shuffle.json` |
| **Figure 4a** — probe AUROC by layer | `02_train_probes.py` | `results/probes/{model}_*.pkl` |
| **Figure 4b** — pairwise cosine of probe directions | `03_geometry_analysis.py` | `results/tables/{model}_geometry.json` |
| **Figure 4c** — asymmetric activation steering at layer 15 | `05_steering_vectors.py` | `results/steering/{model}_*.csv` |
| **Methods** — activation patching localises layer 15 | `04_activation_patching.py` | `results/patching/{model}_patching_summary.json` |
| **All figures** — Plotly violins, brackets, tables | `notebooks/steering_plots.ipynb` | `mechanistic/*.svg` |
| **Methods** — residual-stream extraction (33 layers × 4096 dim) | `01_extract_hidden_states.py` | `results/hidden_states/{model}.h5` |

## Requirements

- Python ≥ 3.10
- CUDA GPU with ≥ 40 GB VRAM (extraction and steering for the 8 B primary model)
- Packages from the top-level `requirements.txt` (the mechanistic-specific block is
  already included there)
- A HuggingFace token with access to `meta-llama/Llama-3.1-8B-Instruct`

## Configuration

1. Accept the Llama 3.1 model licence on HuggingFace.
2. From the repo root, copy and populate `.env`:

   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=hf_...
   ```

3. Inspect `mechanistic/config/config.yaml` if you need to change paths, the probe
   layer thresholds, or the steering alpha sweep. Secrets are **never** read from
   YAML — `src/config.py` will raise if `hf_token:` is present in the YAML.

## Pipeline

All scripts use absolute outputs rooted at the `mechanistic/` directory, derived from
`mechanistic/config/config.yaml` `paths:` section. Run from `mechanistic/`:

```bash
cd mechanistic

# 1. Extract residual-stream activations for Llama3.1:8b   (~25 min on 1× H100)
python 01_extract_hidden_states.py

# 2. Train per-layer logistic probes  (Figure 4a)
python 02_train_probes.py

# 3. Probe geometry / cosines  (Figure 4b)
python 03_geometry_analysis.py

# 4. Activation patching — justifies layer 15 as the steering point  (Methods)
python 04_activation_patching.py

# 5. Asymmetric activation steering  (Figure 4c)
python 05_steering_vectors.py

# 6. Within-tertile shuffle  (Figure 3c)
python 06_shuffle_within_tertile.py

# 7. Cross-tertile shuffle  (Figure 3d)
python 07_cross_tertile_shuffle.py

# 8. Render the figures from the JSONs / CSVs produced above
jupyter notebook notebooks/steering_plots.ipynb
```

Each script supports `--force` to re-run when an output already exists, and
`--model_key` to switch to a comparison model from `config/config.yaml`
(e.g. `--model_key qwen2.5-7b`).

## Output layout

Default paths from `config/config.yaml`:

```
mechanistic/results/
├── hidden_states/   # HDF5: (n_samples, 33 layers, 4096 dim)
├── probes/          # pickled ProbeResult + per-probe AUROC JSON
├── patching/        # per-layer Δ(log-prob) for residual / attn / MLP
├── steering/        # per-alpha CSVs and probe-direction NPY files
├── tables/          # JSON summaries used by the plotting notebook
└── plots/           # PNG/PDF figures emitted by individual scripts
```

All `results/` directories are git-ignored at the repo root.

## Notes

- The `parents[1]` path inside each script assumes the file lives at
  `mechanistic/0N_*.py`. If you move a script, update `ROOT` accordingly.
- The notebook resolves `ROOT = Path("..").resolve()`, i.e. it expects to live at
  `mechanistic/notebooks/`. Outputs are written next to the notebook by default.
- The probe and steering scripts are deterministic given `seed: 42` in the YAML and a
  fixed sklearn `StratifiedKFold(seed=42)`. Exact reproducibility of the steering
  curves additionally requires a single GPU with sequential execution
  (see paper §"Reproducibility analysis under deterministic conditions").
