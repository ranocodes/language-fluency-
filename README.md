# Language Learning Fluency Streamlit App

This app uses the supplied model to estimate whether a learner may drop out. It supports individual predictions and CSV batch predictions. Results include the model's dropout probability and practical suggestions based on simple input thresholds; suggestions are generated locally and do not require an API.

## Run locally

From this folder, install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `language_fluency_model.pkl` and `language_fluency_features.pkl` in this folder beside `app.py`.

## Batch CSV format

Include one row per learner and the following model input columns:

```text
fsi_category,total_study_hours,active_use_share,immersion_months,speaking_practice,feedback_quality,comprehensible_input_hours,uses_srs,motivation,consistency
```

`fsi_category` should be 1–4, `active_use_share` should be between 0 and 1, the 0–10 measures should be between 0 and 10, and hour/month values should be nonnegative. The app requires each input column to be present and numeric.

The model's target is `dropped_out`; it does not predict CEFR level or attained fluency. Predictions and guidance are estimates based on the model and fixed thresholds, respectively.
