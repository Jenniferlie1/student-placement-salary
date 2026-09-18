import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict"

def main():
    st.set_page_config(page_title="Student Placement App", layout="wide")

    st.title("🎓 Placement & Salary Prediction App")
    st.write("Predict student placement and salary using API")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Input
    with st.sidebar.form("form"):
        st.header("Input Data Student")

        gender = st.selectbox("Gender", ["Male", "Female"])

        ssc = st.slider("SSC %", 0, 100, 50)
        hsc = st.slider("HSC %", 0, 100, 50)
        degree = st.slider("Degree %", 0, 100, 50)
        cgpa = st.slider("CGPA", 0.0, 10.0, 5.0)

        entrance = st.slider("Entrance Score", 0, 100, 50)
        tech = st.slider("Technical Skill", 0, 100, 50)
        soft = st.slider("Soft Skill", 0, 100, 50)

        internship = st.number_input("Internship", 0, 10, 0)
        projects = st.number_input("Projects", 0, 20, 0)
        experience = st.number_input("Experience (Months)", 0, 60, 0)

        cert = st.number_input("Certifications", 0, 20, 0)
        attendance = st.slider("Attendance %", 0, 100, 75)
        backlogs = st.number_input("Backlogs", 0, 20, 0)

        extra = st.selectbox("Extracurricular", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    if submit:

        payload = {
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
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            # Output
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Placement Result")
                if result["placement_status"] == 1:
                    st.success("✅ Placed")
                else:
                    st.error("❌ Not Placed")

            with col2:
                st.subheader("Salary Prediction")
                st.info(f"{result['salary_package_lpa']} LPA")

            # Save History
            st.session_state.history.append({**payload, **result})

            # Data Preview
            st.subheader("Input Data")
            st.dataframe(pd.DataFrame([payload]))

            # Visual
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

        else:
            st.error("API Error")

    # History
    if st.session_state.history:
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history))


if __name__ == "__main__":
    main()