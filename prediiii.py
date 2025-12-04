import streamlit as st
import pandas as pd
import plotly.express as px
import json
import io
from fpdf import FPDF
import os

def render_symptom_input(all_symptoms, base_model, decode_label, training, helpers, TMP_DIR, user_id, sel_row):
    """
    Render the Symptom Input module in Streamlit.

    Parameters:
    - all_symptoms: list of all symptom strings.
    - base_model: trained model for prediction.
    - decode_label: function or dict to decode model output to disease names.
    - training: training data or metadata.
    - helpers: optional helper models or data.
    - TMP_DIR: directory path for temporary files.
    - user_id: current user id (for saving predictions).
    - sel_row: selected patient row or None.
    """

    def build_weighted_vector(selected_symptoms_with_severity):
        """
        Build a weighted vector for the model input from symptoms and severities.
        Severity mapping: Mild=0.33, Moderate=0.66, Severe=1.0
        """
        severity_map = {"Mild": 0.33, "Moderate": 0.66, "Severe": 1.0}
        vector = [0.0] * len(all_symptoms)
        symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
        for symptom, severity in selected_symptoms_with_severity.items():
            if symptom in symptom_index:
                vector[symptom_index[symptom]] = severity_map.get(severity, 0)
        return vector

    def topk_predictions(model, feature_vector, decode_label, training, helpers=None, k=5):
        """
        Call the model to get top k predictions with confidence scores.
        Returns a list of tuples (disease_label, confidence_score).
        """
        import numpy as np
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([feature_vector])[0]
        else:
            from scipy.special import softmax
            scores = model.decision_function([feature_vector])[0]
            proba = softmax(scores)
        topk_idx = proba.argsort()[-k:][::-1]
        results = []
        for idx in topk_idx:
            label = decode_label[idx] if isinstance(decode_label, (list, dict)) else decode_label(idx)
            confidence = proba[idx]
            results.append((label, confidence))
        return results

    def save_prediction_to_db(user_id, sel_row, predictions, selected_symptoms_with_severity):
        """
        Save prediction data to a database or file system.
        This is a placeholder - implement as needed.
        """
        # Example: save to a CSV or JSON file in TMP_DIR with user_id and timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{user_id}_{timestamp}.json"
        filepath = os.path.join(TMP_DIR, filename)
        data = {
            "user_id": user_id,
            "patient": sel_row if sel_row is not None else "N/A",
            "predictions": predictions,
            "symptoms": selected_symptoms_with_severity,
            "timestamp": timestamp
        }
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            st.success(f"Prediction saved to {filepath}")
        except Exception as e:
            st.error(f"Failed to save prediction: {e}")

    def generate_json_report(predictions, selected_symptoms_with_severity):
        """
        Generate JSON report string.
        """
        report = {
            "predictions": [{"disease": d, "confidence": float(c)} for d, c in predictions],
            "symptoms": selected_symptoms_with_severity
        }
        return json.dumps(report, indent=2)

    def generate_pdf_report(predictions, selected_symptoms_with_severity):
        """
        Generate PDF report bytes.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "Disease Prediction Report", ln=True, align="C")
        pdf.ln(5)

        pdf.cell(0, 10, "Predictions:", ln=True)
        for disease, conf in predictions:
            pdf.cell(0, 10, f"- {disease}: {conf:.2%}", ln=True)

        pdf.ln(5)
        pdf.cell(0, 10, "Symptoms and Severities:", ln=True)
        for symptom, severity in selected_symptoms_with_severity.items():
            pdf.cell(0, 10, f"- {symptom}: {severity}", ln=True)

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return pdf_bytes

    st.header("Symptom Input & Disease Prediction")

    # 1. Symptom selection via multiselect
    selected_symptoms = st.multiselect(
        "Select symptoms from the list",
        options=all_symptoms,
        key="symptom_multiselect"
    )

    # 2. Optional manual symptom input
    manual_symptoms_input = st.text_input(
        "Or type symptoms manually (comma-separated)",
        key="manual_symptoms_input"
    )
    manual_symptoms = []
    if manual_symptoms_input.strip():
        manual_symptoms = [s.strip() for s in manual_symptoms_input.split(",") if s.strip()]
    # Combine and deduplicate symptoms
    combined_symptoms = list(dict.fromkeys(selected_symptoms + manual_symptoms))

    if not combined_symptoms:
        st.warning("Please select or enter at least one symptom to proceed.")
        return

    # 3. Severity slider for each symptom
    severity_options = ["Mild", "Moderate", "Severe"]
    selected_symptoms_with_severity = {}
    st.subheader("Set severity for each symptom")
    for idx, symptom in enumerate(combined_symptoms):
        severity = st.select_slider(
            f"Severity for '{symptom}'",
            options=severity_options,
            value="Moderate",
            key=f"severity_slider_{idx}"
        )
        selected_symptoms_with_severity[symptom] = severity

    # 5. Predict button
    if st.button("Predict", key="predict_button"):
        try:
            # Build feature vector
            feature_vector = build_weighted_vector(selected_symptoms_with_severity)

            # Get top predictions
            predictions = topk_predictions(base_model, feature_vector, decode_label, training, helpers)

            if not predictions:
                st.warning("No predictions returned from the model.")
                return

            # Display predictions in a table
            st.success("Predictions generated successfully!")
            df_preds = pd.DataFrame(predictions, columns=["Disease", "Confidence"])
            df_preds["Confidence"] = df_preds["Confidence"].apply(lambda x: f"{x:.2%}")
            st.table(df_preds)

            # 6. Recommendations expander
            st.subheader("Recommendations")
            for disease, _ in predictions:
                rec = training.get(disease, {})
                with st.expander(f"Recommendations for {disease}"):
                    desc = rec.get("Description", "No description available.")
                    prec = rec.get("Precautions", [])
                    meds = rec.get("Medications", [])
                    diet = rec.get("Diet", [])
                    workouts = rec.get("Workouts", [])

                    st.markdown(f"**Description:**\n{desc}")
                    if prec:
                        st.markdown("**Precautions:**")
                        for p in prec:
                            st.markdown(f"- {p}")
                    if meds:
                        st.markdown("**Medications:**")
                        for m in meds:
                            st.markdown(f"- {m}")
                    if diet:
                        st.markdown("**Diet:**")
                        for d in diet:
                            st.markdown(f"- {d}")
                    if workouts:
                        st.markdown("**Workouts:**")
                        for w in workouts:
                            st.markdown(f"- {w}")

            # 7. Visualization of model probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Disease": [d for d, _ in predictions],
                "Confidence": [c for _, c in predictions]
            })
            fig = px.bar(
                prob_df,
                x="Confidence",
                y="Disease",
                orientation="h",
                labels={"Confidence": "Confidence", "Disease": "Disease"},
                text=prob_df["Confidence"].apply(lambda x: f"{x:.2%}")
            )
            fig.update_layout(yaxis={"categoryorder":"total ascending"}, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # 8. Optionally save predictions if patient selected
            if sel_row is not None:
                save_prediction_to_db(user_id, sel_row, predictions, selected_symptoms_with_severity)

            # 9. Export buttons
            st.subheader("Export Report")

            json_report = generate_json_report(predictions, selected_symptoms_with_severity)
            st.download_button(
                label="Download JSON Report",
                data=json_report,
                file_name="disease_prediction_report.json",
                mime="application/json",
                key="download_json"
            )

            pdf_report = generate_pdf_report(predictions, selected_symptoms_with_severity)
            st.download_button(
                label="Download PDF Report",
                data=pdf_report,
                file_name="disease_prediction_report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
