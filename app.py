import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👥",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load("employee_turnover_model.pkl")

try:
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
except Exception as exc:
    st.error("Could not load employee_turnover_model.pkl. Make sure it is in the same folder as app.py.")
    st.exception(exc)
    st.stop()

st.title("👥 Employee Turnover Prediction")
st.write(
    "Enter the employee's feature values to estimate whether the model predicts "
    "that the employee will leave or stay."
)
st.warning(
    "The provided dataset contains normalized values (approximately 0 to 1). "
    "Enter values on that same scale. This is an educational project, not a "
    "validated HR decision tool."
)

original_features = [
    "Job_Satisfaction",
    "Performance_Rating",
    "Years_At_Company",
    "Work_Life_Balance",
    "Distance_From_Home",
    "Monthly_Income",
    "Education_Level",
    "Age",
    "Num_Companies_Worked",
    "Employee_Role",
    "Annual_Bonus",
    "Training_Hours",
    "Department",
]

labels = {
    "Job_Satisfaction": "Job Satisfaction",
    "Performance_Rating": "Performance Rating",
    "Years_At_Company": "Years at Company",
    "Work_Life_Balance": "Work-Life Balance",
    "Distance_From_Home": "Distance from Home",
    "Monthly_Income": "Monthly Income",
    "Education_Level": "Education Level",
    "Age": "Age",
    "Num_Companies_Worked": "Number of Previous Companies",
    "Employee_Role": "Employee Role (encoded)",
    "Annual_Bonus": "Annual Bonus",
    "Training_Hours": "Training Hours",
    "Department": "Department (encoded)",
}

with st.form("employee_form"):
    st.subheader("Employee details")
    st.caption("All values use the same normalized scale as the supplied dataset.")
    values = {}
    cols = st.columns(2)

    for idx, feature in enumerate(original_features):
        with cols[idx % 2]:
            values[feature] = st.slider(
                labels[feature],
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=feature,
            )

    submitted = st.form_submit_button("Predict Turnover", type="primary")

if submitted:
    # Re-create engineered features exactly as represented in the dataset.
    values["Annual_Bonus_Squared"] = values["Annual_Bonus"] ** 2
    values["Annual_Bonus_Training_Hours_Interaction"] = (
        values["Annual_Bonus"] * values["Training_Hours"]
    )

    try:
        input_df = pd.DataFrame(
            [[values.get(feature, 0.0) for feature in feature_names]],
            columns=feature_names,
        )
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction result")
        if probabilities is not None:
            leaving_probability = float(probabilities[1])
            st.metric("Model-estimated probability of leaving", f"{leaving_probability:.2%}")
            st.progress(max(0.0, min(1.0, leaving_probability)))

        if prediction == 1:
            st.error("Prediction: Employee may leave the company.")
        else:
            st.success("Prediction: Employee may stay with the company.")

        st.caption(
            "This prediction depends on the model and the entered values. "
            "It should not be used alone to make employment decisions."
        )
    except Exception as exc:
        st.error("Prediction failed. Check that the model file matches this notebook and dataset.")
        st.exception(exc)
