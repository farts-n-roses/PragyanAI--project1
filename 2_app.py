import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hiring Funnel Analytics", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/pragyanaischool/VTU_Internship_DataSets/refs/heads/main/student_data_placement_interview_funnel_analysis_project_10.csv"
    df = pd.read_csv(url) 
    return df

df = load_data()
df["Failed_Stage"] = df["Failed_Stage"].str.strip()

df_encoded = df.copy()
df_encoded["Failed_Stage"] = df_encoded["Failed_Stage"].map({
    "Round1": 1,
    "Round2": 2,
    "Round3": 3,
    "Selected": 4
})
numeric_df = df_encoded.select_dtypes(include="number")
corr = numeric_df.corr(method="spearman")

st.title("Hiring Funnel Analytics Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Funnel Analysis",
    "Feature Impact",
    "Failure Analysis",
    "Key Insights",
    "Interactive 'What-If' Simulator"
])

# -----------------------------
# PRECOMPUTED FUNNEL
# -----------------------------
funnel = {
    "Applied": 102876,
    "Shortlisted": 92693,
    "Interview": 68922,
    "Offer": 33806,
    "Joined": 10212
}

stages = list(funnel.keys())
values = list(funnel.values())

# =============================
# OVERVIEW
# =============================
if page == "Overview":
    st.header("🏠 Overview")

    col1, col2, col3, col4 = st.columns(4)

    # col1.metric("Applied", funnel["Applied"])
    # col2.metric("Shortlisted", funnel["Shortlisted"])
    # col3.metric("Interviewed", funnel["Interview"])
    # col4.metric("Joined", funnel["Joined"])
    
    col1.markdown("### 📥 Applied")
    col1.metric("", funnel["Applied"])

    col2.markdown("### 🧾 Shortlisted")
    col2.metric("", funnel["Shortlisted"])

    col3.markdown("### 🎤 Interview")
    col3.metric("", funnel["Interview"])

    col4.markdown("### 🎉 Joined")
    col4.metric("", funnel["Joined"])

    drop = (1 - funnel["Joined"]/funnel["Applied"])*100
    st.metric("Overall Drop-off %:", f"{drop:.2f}%")

    st.markdown("### Key Insight")
    # st.info("Most attrition happens in Interview → Offer and Offer → Join stages.")
    st.success("""💡 Most attrition(drop-off) happens after interviews (Interview → Offer stage) and especially after offers (Offer → Join stage).
    This suggests evaluation + retention issues, not applicant quality.
    """)

# =============================
# FUNNEL ANALYSIS
# =============================
elif page == "Funnel Analysis":
    st.header("📊 Funnel Analysis")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(stages, values, marker='o', linewidth=3)
    for i, value in enumerate(values):
        ax.text(i, value, str(value), ha='center', va='bottom')
    ax.set_title("Hiring Funnel Drop-off", fontsize=16)
    ax.set_ylabel("Candidates")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    survival = np.array(values)/values[0]

    percent = np.array(values) / values[0] * 100
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(stages, percent, marker='o', linewidth=3)
    for i, v in enumerate(percent):
        ax2.text(stages[i], v+1, f"{v:.1f}%", ha='center')
    ax2.set_title("Survival Rate (%) Across Funnel")
    ax2.set_ylabel("Percentage")
    st.pyplot(fig2)

    # fig2, ax2 = plt.subplots()
    # ax2.plot(stages, survival, marker='o')
    # ax2.set_title("Survival Rate Across Funnel")
    # ax2.set_ylim(0,1)
    # st.pyplot(fig2)

    st.subheader("📉 Stage-wise Drop Rates")
    for i in range(len(stages)-1):
        drop = (1 - values[i+1]/values[i])*100
        st.write(f"{stages[i]} → {stages[i+1]}: {drop:.2f}% drop")

# =============================
# FEATURE IMPACT
# =============================
elif page == "Feature Impact":
    st.header("🎯 Feature Impact Analysis")

    if "CGPA" in df.columns:
        st.subheader("CGPA vs Outcome")
        st.write(df.groupby("Failed_Stage")["CGPA"].mean())

    if "Projects" in df.columns:
        st.subheader("Projects vs Outcome")
        st.write(df.groupby("Failed_Stage")["Projects"].mean())

    st.subheader("Correlation Heatmap")

    # numeric_cols = df.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

# =============================
# FAILURE ANALYSIS
# =============================
elif page == "Failure Analysis":
    st.header("🔥 Failure Stage Analysis")

    if "Failed_Stage" in df.columns:
        st.subheader("Failure Distribution")
        st.write(df["Failed_Stage"].value_counts(normalize=True)*100)

    st.subheader("Stage-wise Insight")
    st.info("""
    - Round 1 is the biggest elimination stage  
    - Interview → Offer is the biggest bottleneck  
    - Offer → Join has extreme leakage  
    """)

# =============================
# KEY INSIGHTS
# =============================
elif page == "Key Insights":
    st.header("💡 Key Insights")

    st.markdown("""
    ### 📌 Summary of Findings

    - CGPA has negligible impact on outcomes  
    - Project count does not differentiate success  
    - Major attrition occurs in:
        - Interview → Offer  
        - Offer → Join  

    ### 🚨 Core Bottleneck
    The system is not limited by applicant quality, but by:
    - evaluation consistency
    - post-offer retention

    ### 📌 Executive Summary
    - Biggest drop: Interview → Offer (~51%)
    - Worst conversion: Offer → Join (~70%)
    - CGPA & Projects show weak correlation
    - Problem is NOT student quality, but evaluation & retention

    ### 📊 Final Interpretation
    The hiring process is structurally front-loaded with screening efficiency, but suffers from late-stage inefficiencies.
    """)
 
# ===============================
# INTERACTIVE 'WHAT-IF' SIMULATOR
# ===============================
elif page == "Interactive 'What-If' Simulator":
    st.header("🔮 What-If Simulator")

    improve = st.sidebar.slider("Improve Offer → Join Conversion (%)", 0, 100, 30)
    
    sim_joined = funnel["Offer"] * (improve / 100)

    st.write("### Simulated Joined Candidates")
    st.metric("Joined", int(sim_joined))
    
    if improve <= 30:
        st.info(f"""If Offer → Join is {improve}%, joined candidates decrease from {funnel['Joined']} → {int(sim_joined)}.""")
    else:
        st.info(f"""If Offer → Join improves to {improve}%, joined candidates increase from {funnel['Joined']} → {int(sim_joined)}.""")
    

# -----------------------------------------------------------------------------------------------------------------------------------
