from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "language_fluency_model.pkl"
FEATURES_PATH = BASE_DIR / "language_fluency_features.pkl"

st.set_page_config(page_title="Language Learning Fluency", page_icon="🗣️", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, list(features)


def guidance_for(row):
    suggestions = []
    if row.get("active_use_share", 0) < 0.4:
        suggestions.append("Shift some study time from passive review to active speaking, writing, and conversation.")
    if row.get("speaking_practice", 0) < 5:
        suggestions.append("Schedule regular speaking sessions, even short ones, to build a sustainable practice habit.")
    if row.get("feedback_quality", 0) < 5:
        suggestions.append("Add reliable corrective feedback, such as a tutor, teacher, or language exchange partner.")
    if row.get("consistency", 0) < 5:
        suggestions.append("Set a small daily or weekly study target and track completed sessions.")
    if row.get("immersion_months", 0) < 1:
        suggestions.append("Create more contact with the language through media, conversation groups, or community activities.")
    if row.get("comprehensible_input_hours", 0) < 100:
        suggestions.append("Build a steady routine of listening and reading material that is understandable but still challenging.")
    if row.get("motivation", 0) < 5:
        suggestions.append("Choose personally relevant topics and set short milestones to make progress visible.")
    if row.get("total_study_hours", 0) < 100:
        suggestions.append("Keep accumulating study time with a manageable routine; consistent practice matters over time.")
    if row.get("uses_srs", 0) == 0:
        suggestions.append("Consider spaced repetition for vocabulary and phrases that are difficult to retain.")
    if not suggestions:
        suggestions.append("Keep the current balanced routine and review progress regularly so the plan stays sustainable.")
    return suggestions


@st.cache_data
def predict(_model, frame):
    # Streamlit ignores underscore-prefixed arguments when building the cache key.
    probabilities = _model.predict_proba(frame)[:, 1]
    predictions = _model.predict(frame)
    return predictions, probabilities


def prediction_label(prediction):
    return "Higher dropout risk" if int(prediction) == 1 else "Lower dropout risk"


def show_result(row, prediction, probability):
    st.subheader(prediction_label(prediction))
    st.metric("Estimated dropout probability", f"{probability:.1%}")
    if int(prediction) == 1:
        st.warning("The model classifies this learner as at risk of dropping out.")
    else:
        st.success("The model does not classify this learner as at risk of dropping out.")
    st.caption("This is a model estimate based on the training data, not a certainty or a measure of language ability.")
    st.markdown("**Suggested actions**")
    for suggestion in guidance_for(row):
        st.markdown(f"- {suggestion}")


st.title("Language Learning Fluency")
st.write("Estimate learner dropout risk and identify practical ways to support continued progress.")

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.error("Model files are missing. Keep the model and feature pickle files beside app.py.")
    st.stop()

try:
    model, features = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model artifacts: {exc}")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not provide probability estimates, which this app requires.")
    st.stop()

st.caption("The prediction target is dropout risk. Guidance is generated from simple input thresholds; it does not use an external API.")
single_tab, batch_tab = st.tabs(["One learner", "CSV batch"])

with single_tab:
    st.subheader("Learner details")
    st.write("Enter the learner's current study and practice information.")
    with st.form("single_prediction"):
        values = {}
        columns = st.columns(2)
        for index, feature in enumerate(features):
            container = columns[index % 2]
            with container:
                if feature == "fsi_category":
                    values[feature] = st.selectbox(
                        "Target language difficulty (FSI category)",
                        options=[1, 2, 3, 4],
                        format_func=lambda value: {
                            1: "1 — Easier for English speakers",
                            2: "2 — Generally easier",
                            3: "3 — More difficult",
                            4: "4 — Most difficult (for example, Mandarin or Arabic)",
                        }[value],
                    )
                elif feature == "active_use_share":
                    values[feature] = st.slider("Active use share", 0.0, 1.0, 0.4, 0.05, help="Fraction of study that involves active use or output.")
                elif feature in {"immersion_months"}:
                    values[feature] = st.number_input("Immersion months", min_value=0.0, value=0.0, step=1.0)
                elif feature in {"speaking_practice", "feedback_quality", "motivation", "consistency"}:
                    names = {
                        "speaking_practice": "Speaking practice (0–10)",
                        "feedback_quality": "Feedback quality (0–10)",
                        "motivation": "Motivation (0–10)",
                        "consistency": "Consistency (0–10)",
                    }
                    values[feature] = st.slider(names[feature], 0.0, 10.0, 5.0, 0.5)
                elif feature == "uses_srs":
                    values[feature] = int(st.checkbox("Uses spaced repetition (SRS)"))
                else:
                    labels = {
                        "total_study_hours": "Total study hours",
                        "comprehensible_input_hours": "Comprehensible input hours",
                    }
                    values[feature] = st.number_input(labels.get(feature, feature.replace("_", " ").title()), min_value=0, value=100, step=10)
        submitted = st.form_submit_button("Estimate dropout risk", type="primary")

    if submitted:
        input_frame = pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)
        preds, probs = predict(model, input_frame)
        show_result(values, preds[0], probs[0])

with batch_tab:
    st.subheader("Upload learner records")
    st.write("Upload a CSV with one row per learner and a column for every model input feature.")
    st.code(",".join(features), language="text")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif batch.empty:
                st.error("The uploaded CSV has no learner rows.")
            else:
                model_input = batch.loc[:, features].apply(pd.to_numeric, errors="coerce")
                invalid_rows = model_input.index[model_input.isna().any(axis=1)].tolist()
                if invalid_rows:
                    st.error("Some required values are blank or non-numeric. Check CSV rows (data row numbers): " + ", ".join(str(index + 2) for index in invalid_rows[:20]))
                else:
                    preds, probs = predict(model, model_input)
                    results = batch.copy()
                    results["dropout_prediction"] = [prediction_label(value) for value in preds]
                    results["dropout_probability"] = probs
                    results["dropout_probability_percent"] = (probs * 100).round(1)
                    results["suggested_actions"] = model_input.apply(
                        lambda row: " ".join(guidance_for(row.to_dict())), axis=1
                    )
                    st.dataframe(results, use_container_width=True)
                    st.download_button(
                        "Download predictions as CSV",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="language_fluency_predictions.csv",
                        mime="text/csv",
                    )
                    st.caption("The downloaded CSV includes individualized rule based suggestions for every learner.")
        except Exception as exc:
            st.error(f"Could not read or process this CSV: {exc}")
