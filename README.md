# Sabeeq | سَبيق

**An AI system that predicts where a threat will land, recommends protective actions, and lets the power network heal itself — built as a simulation on synthetic data.**

> **Designed and developed by Aseel** ([@aseel-aihub](https://github.com/aseel-aihub))
> © 2026 Aseel. All rights reserved — see [LICENSE](LICENSE).
>
> Entered in the **SAIF** competition (Global Security & Technology Summit / Black Hat MEA). Originally proposed as a PSDSARC graduation project.

---

## Scope

- All coordinates, facilities and tracks are **fictional**.
- No real sensors, no real installations, no classified data.
- The system only gives **advisory text**; it does not control anything.

---

## What it does

| # | Feature | File |
|---|---|---|
| 1 | Predicts the flight path and impact point (with air drag) | `src/ballistics.py`, `src/geo.py` |
| 2 | Finds which facility is in danger | `src/facilities.py` |
| 3 | AI model predicts the target sector + confidence | `src/pattern_model.py` |
| 4 | Explainable protective recommendations | `src/response_rules.py` |
| 5 | Self-healing power network (Dijkstra) | `src/self_healing.py` |

**Tech stack:** Python · Streamlit · Pandas / NumPy · Scikit-learn · NetworkX · Plotly

---

## How to run

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate the synthetic data
python scripts/generate_dataset.py

# 3. Train the model
python scripts/train_model.py

# 4. Run the dashboard
streamlit run app.py

# 5. Run the tests
pytest -v
```

> Steps 2 and 3 are optional: if the model is missing, the app trains it automatically on first run.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **86.0%** |
| Majority baseline (guessing) | 37.8% |
| Improvement over guessing | **+48.2 points** |
| Macro F1 | 0.842 |
| 5-fold cross-validation | 83.8% ± 2.0 |

In my first version the labels were random, so the model scored 40.0% while guessing scored 40.3% — it learned nothing. I fixed it by making every label come from the physics (launch → trajectory → impact point → nearest facility → its sector).

The accuracy is kept below 100% on purpose with 8% label noise (like real sensor errors).

---

## Security & quality

- **Model integrity check:** the saved model has a SHA-256 fingerprint. If the file is changed, it is not loaded and a clean model is trained instead (loading a modified `.joblib` file can run harmful code).
- **Input validation:** NaN, infinity, text and out-of-range values are rejected with clear errors.
- **No HTML injection:** all names are escaped before being shown as HTML.
- **Safe data loading:** a broken or edited `facilities.csv` is detected and the built-in list is used instead.
- **Hidden error details:** users see a friendly message, not technical tracebacks.
- **55 automated tests** (`pytest`).

---

## Project structure

```
Sabeeq/
├── app.py                    # Streamlit dashboard (4 tabs)
├── config.py                 # All settings
├── data/                     # Synthetic data
├── models/                   # Trained model (auto-generated)
├── src/
│   ├── geo.py                # Map math
│   ├── ballistics.py         # Trajectory with drag
│   ├── facilities.py         # Facilities + nearest one
│   ├── data_generation.py    # Dataset generator
│   ├── pattern_model.py      # Random Forest
│   ├── response_rules.py     # Recommendation rules
│   └── self_healing.py       # Self-healing network
├── scripts/                  # Generate data / train model
├── tests/test_core.py        # 55 tests
└── LICENSE                   # All rights reserved
```

---

## Limitations

- The object is assumed to not change direction; no wind and constant drag.
- The Earth is treated as a sphere (~0.5% error at these distances).
- The model learns a physical relationship, not real attacker intent.
- The network model checks if a path exists, not if it can carry the full load.

---

## Author

**Aseel** — [github.com/aseel-aihub](https://github.com/aseel-aihub)

© 2026 Aseel. All rights reserved. No part of this project (code or idea) may be copied or reused without written permission.
