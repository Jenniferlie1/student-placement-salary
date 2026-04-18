import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load Model (.pkl)
clf = joblib.load("artifacts/classification_model.pkl")
reg = joblib.load("artifacts/regression_model.pkl")


def main():
    st.set_page_config(page_title="Student Placement App", layout="wide")

    st.title("🎓 Placement & Salary Prediction App")
    st.write("Predict student placement status and estimate salary.")

    # Init History
    if "history" not in st.session_state:
        st.session_state.history = []

    # Form
    with st.sidebar.form("input_form"):
        st.header("Input Data Student")

        gender = st.selectbox("Gender", ["Male", "Female"])

        ssc = st.slider("SSC %", 0, 100, 50)
        hsc = st.slider("HSC %", 0, 100, 50)
        degree = st.slider("Degree %", 0, 100, 50)
        cgpa = st.slider("CGPA", 0.0, 10.0, 5.0)

        entrance = st.slider("Entrance Exam Score", 0, 100, 50)
        tech = st.slider("Technical Skill", 0, 100, 50)
        soft = st.slider("Soft Skill", 0, 100, 50)

        internship = st.number_input("Internship Count", 0, 10, 0)
        projects = st.number_input("Live Projects", 0, 20, 0)
        experience = st.number_input("Experience (Months)", 0, 60, 0)

        cert = st.number_input("Certifications", 0, 20, 0)
        attendance = st.slider("Attendance %", 0, 100, 75)
        backlogs = st.number_input("Backlogs", 0, 20, 0)

        extra = st.selectbox("Extracurricular", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    # Prediction
    if submit:

        result_class, result_salary, input_df = make_prediction(
            gender, ssc, hsc, degree, cgpa,
            entrance, tech, soft,
            internship, projects, experience,
            cert, attendance, backlogs, extra
        )

        # Save to History
        st.session_state.history.append({
            "Gender": gender,
            "SSC %": ssc,
            "HSC %": hsc,
            "Degree %": degree,
            "CGPA": cgpa,
            "Entrance Score": entrance,
            "Technical Skill": tech,
            "Soft Skill": soft,
            "Internship Count": internship,
            "Live Projects": projects,
            "Experience (Months)": experience,
            "Certifications": cert,
            "Attendance %": attendance,
            "Backlogs": backlogs,
            "Extracurricular": extra,
            "Placement": "Placed" if result_class == 1 else "Not Placed",
            "Salary (LPA)": round(result_salary, 2)
        })

        # Output
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Placement Result")
            if result_class == 1:
                st.success("✅ Placed")
            else:
                st.error("❌ Not Placed")

        with col2:
            st.subheader("Salary Prediction")
            st.info(f"{result_salary:.2f} LPA")

        # Data Preview
        st.subheader("Input Data")
        st.dataframe(input_df)

        # Visualization
        st.subheader("Skill Analysis")

        skills = ["Technical Skill", "Soft Skill"]
        values = [tech, soft]
        colors = ["#A8D5BA", "#AFCBFF"]

        plt.style.use("seaborn-v0_8")

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(skills, values, color=colors, edgecolor="gray")


        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1,
                    f"{yval:.0f}", ha='center', fontweight='bold')

        ax.set_ylim(0, 100)
        ax.set_title("Skill Comparison", fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig)

    # History
    if st.session_state.history:
        st.subheader("Prediction History")

        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)

# Inference
def make_prediction(
    gender, ssc, hsc, degree, cgpa,
    entrance, tech, soft,
    internship, projects, experience,
    cert, attendance, backlogs, extra
):

    input_df = pd.DataFrame([{
        "gender": gender,
        "ssc_percentage": ssc,
        "hsc_percentage": hsc,
        "degree_percentage": degree,
        "cgpa": cgpa,
        "entrance_exam_score": entrance,
        "technical_skill_score": tech,
        "soft_skill_score": soft,
        "internship_count": internship,
        "live_projects": projects,
        "work_experience_months": experience,
        "certifications": cert,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "extracurricular_activities": extra
    }])

    pred_class = clf.predict(input_df)[0]
    pred_salary = reg.predict(input_df)[0]

    return pred_class, pred_salary, input_df


if __name__ == "__main__":
    main()