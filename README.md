# A band-observability criterion for sensor placement in continuous data assimilation of a chaotic two-dimensional flow

Reference code for the paper. Continuous data assimilation by nudging is known to converge when *enough*
observations are supplied, but network design is a question of *placement*; we show the controlling quantity
is the observability of the **determining band** `δ_K` — not the sensor count and not the coverage radius — and
that for periodic sampling the threshold is provable: `R` equidistant sensor rows leave a kernel of dimension
`(2K+1)(2K+1−R)` on the band, closed exactly by the Shepard interpolant.

## Layout

```
├── nolab/           # shared library
├── script/          # frozen experiment scripts
├── environment.yml  # conda environment
├── requirements.txt # pip equivalent
└── README.md
```

## Install

```bash
conda env create -f environment.yml
conda activate neuralobserver
```

Only `script/35_learned_gain.py` needs torch; everything else uses numpy/scipy/matplotlib.

## Run

```bash
python script/45_band_gram_spectrum.py   # Proposition: kernel dimension 33/22/11/0 (seconds)
python script/48_tail_rebound_audit.py   # tail-half rebound audit (reads stored curves, ~2 s)
python script/41_stripe_nyquist_v3.py    # fixed-count stripe Nyquist (~100 min)
```

Truth fields are cached in a `truths/` directory next to the repository and regenerated automatically by
`nolab.get_truth` when missing. Result files are written to `results/<name>/results.json` with solver
provenance.

## Headline numbers

- `δ_K` ranks the plateau with Kendall `τ = 0.71` / `AUC = 0.98`; coverage radius gives `0.52` / `0.88`.
- No layout with `δ_K ≤ 0` synchronises (0 / 40).
- Kernel dimension `(2K+1)(2K+1−R)` verified as `33 / 22 / 11 / 0` at `R = 8 / 9 / 10 / 11`, `K_c = 5`.
- No counterexample among the 22-row fixed-`n = 784` ladder at `T = 8`, threshold moving `9 → 11 → 15`.
- nudging vs LETKF rate `1.21` vs `2.60` at 1/32 solver cost.

## License

MIT.
