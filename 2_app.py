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

# CLEAN DATA
df = df[
    (df["Joined"] <= df["Offer_Received"]) &
    (df["Offer_Received"] <= df["Interview_Attended"]) &
    (df["Interview_Attended"] <= df["Shortlisted"]) &
    (df["Shortlisted"] <= df["Applied"])
].copy()

df["Failed_Stage"] = df["Failed_Stage"].fillna("").str.strip()
df_encoded = df.copy()
df_encoded["Failed_Stage"] = df_encoded["Failed_Stage"].map({
    "Round1": 1,
    "Round2": 2,
    "Round3": 3,
    "Selected": 4
}).fillna(0)
numeric_df = df_encoded.select_dtypes(include="number")
corr = numeric_df.corr(method="spearman")

df["Internships"] = df["Internships"].fillna(0).astype(int)
df["Job_Role"] = df["Job_Role"].str.strip()
df["Domain"] = df["Domain"].str.strip()
df["Company_Tier"] = df["Company_Tier"].str.strip()

st.title("📊 Hiring Funnel Analytics Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Funnel Analysis",
    "Feature Impact",
    "Failure Analysis",
    "Advanced Analysis",
    "Insights",
    "Interactive 'What-If' Simulator"
])

# -----------------------------
# PRECOMPUTED FUNNEL
# -----------------------------
funnel = {
    "Applied": df["Applied"].sum(),
    "Shortlisted": df["Shortlisted"].sum(),
    "Interview": df["Interview_Attended"].sum(),
    "Offer": df["Offer_Received"].sum(),
    "Joined": df["Joined"].sum()
}

stages = list(funnel.keys())
values = list(funnel.values())

# =============================
# OVERVIEW
# =============================
if page == "Overview":
    st.header("🏠 Overview")

    col1, col2, col3, col4 = st.columns(4)
    
    col1.markdown("### 📥 Applied")
    col1.metric("", funnel["Applied"], "*")

    col2.markdown("### 🧾 Shortlisted")
    col2.metric("", funnel["Shortlisted"])

    col3.markdown("### 🎤 Interview")
    col3.metric("", funnel["Interview"])

    col4.markdown("### 🎉 Joined")
    col4.metric("", funnel["Joined"])

    st.markdown("*Data cleaned from 200K student dataset to remove inconsistent funnel transitions")
    
    st.subheader("🧩 Candidate Segments")

    not_shortlisted = df[df["Shortlisted"] == 0]
    interview_failed = df[
        # (df["Shortlisted"] == 1) &
        (df["Interview_Attended"] == 1) &
        (df["Offer_Received"] == 0)
    ]
    not_placed = df[
        (df["Offer_Received"] == 1) &
        (df["Joined"] == 0)
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Not Shortlisted", len(not_shortlisted))
    col2.metric("Interview Failures", len(interview_failed))
    col3.metric("Placement Offer Declined", len(not_placed))
    
    drop = (1 - funnel["Joined"]/max(funnel["Applied"], 1)) * 100
    st.metric("Overall Drop-off %", f"{drop:.2f}%")

    st.markdown("### Key Insight")
    # st.info("Most attrition happens in Interview → Offer and Offer → Join stages.")
    st.success("""💡 Most attrition(drop-off) happens after interviews (Interview → Offer stage) and especially after offers (Offer → Join stage).
    This suggests evaluation + retention issues, not applicant quality.
    """)

# =============================
# FUNNEL ANALYSIS
# =============================
elif page == "Funnel Analysis":
    st.header("Funnel Analysis")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(stages, values, marker='o', linewidth=3)
    for i, value in enumerate(values):
        ax.text(i, value, str(value), ha='center', va='bottom')
    ax.set_title("Hiring Funnel Drop-off", fontsize=16)
    ax.set_ylabel("Candidates")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    survival = np.array(values)/max(values[0], 1)

    percent = np.array(values) / max(values[0], 1) * 100
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
    drop_data = []
    for i in range(len(stages)-1):
        drop = (1 - values[i+1]/max(values[i], 1)) * 100
        drop_data.append({
            "Stage Transition": f"{stages[i]} → {stages[i+1]}",
            "Drop (%)": f"{drop:.2f}%"
        })
    drop_df = pd.DataFrame(drop_data)
    st.table(drop_df)

# =============================
# FEATURE IMPACT
# =============================
elif page == "Feature Impact":
    st.header("🎯 Feature Impact Analysis")

    if "CGPA" in df.columns:
        st.subheader("CGPA vs Outcome")
        st.write(df_encoded.groupby("Failed_Stage")["CGPA"].mean())

    if "Projects" in df.columns:
        st.subheader("Projects vs Outcome")
        st.write(df_encoded.groupby("Failed_Stage")["Projects"].mean())

    st.subheader("Correlation Heatmap")

    # numeric_cols = df.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=False, linewidths=0.5)
    st.pyplot(fig)

# =============================
# FAILURE ANALYSIS
# =============================
elif page == "Failure Analysis":
    st.header("🔥 Failure Stage Analysis")

    if "Failed_Stage" in df.columns:
        st.subheader("Failure Distribution")
        st.write(df[df["Failed_Stage"] != ""]["Failed_Stage"].value_counts(normalize=True)*100)

    st.subheader("🧠 Round Type Distribution")
    round_types = pd.concat([
        df["Round_1"],
        df["Round_2"],
        df["Round_3"]
    ]).dropna()
    st.bar_chart(round_types.value_counts())
    
    st.subheader("Stage-wise Insight")
    st.info("""
    - Round 1 is the biggest elimination stage  
    - Interview → Offer is the biggest bottleneck  
    - Offer → Join has extreme leakage  
    """)

# =============================
# ADVANCED ANALYSIS
# =============================
elif page == "Advanced Analysis":
    st.header("🚀 Advanced Placement Intelligence")
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    interview_success = df["Offer_Received"].sum() / max(df["Interview_Attended"].sum(), 1)
    offer_join = df["Joined"].sum() / max(df["Offer_Received"].sum(), 1)
    col1.metric("Interview → Offer", f"{interview_success:.2%}")
    col2.metric("Offer → Join", f"{offer_join:.2%}")
    overall = df["Joined"].sum() / max(df["Applied"].sum(), 1)
    col3.metric("Overall Success", f"{overall:.2%}")
    st.markdown("---")
    
    # Role-Wise Analysis
    st.subheader("💼 Role-wise Success Rate")
    role_success = df.groupby("Job_Role")["Joined"].mean().sort_values()
    fig1, ax1 = plt.subplots()
    role_success.plot(kind="barh", ax=ax1)
    ax1.set_xlabel("Success Rate")
    ax1.set_title("Easiest → Hardest Roles")
    st.pyplot(fig1)
    st.caption("Lower success rate = harder role")
    st.info("Higher value = easier role to get offers")

    # Domain Analysis
    st.subheader("🧠 Domain-wise Failure Distribution")
    domain_fail = df[df["Failed_Stage"] != ""].groupby("Domain")["Failed_Stage"].value_counts(normalize=True).unstack().fillna(0)
    fig2, ax2 = plt.subplots()
    sns.heatmap(domain_fail, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    st.caption("Shows where each domain struggles most")
    st.info("GenAI / AI roles typically show higher failure rates")

    # Internship Impact
    st.subheader("🏢 Internship Impact on Offers")
    internship_impact = df.groupby("Internships")["Offer_Received"].mean()
    fig3, ax3 = plt.subplots()
    ax3.plot(internship_impact.index, internship_impact.values, marker='o')
    for i, v in enumerate(internship_impact.values):
        ax3.text(internship_impact.index[i], v, f"{v:.2f}", ha='center')
    ax3.set_title("Internships vs Offer Rate")
    ax3.set_xlabel("Number of Internships")
    ax3.set_ylabel("Offer Rate")
    st.pyplot(fig3)
    st.info("More internships → higher chance of getting offers")

    # Company Tier Analysis
    st.subheader("🏢 Company Tier Difficulty")
    company_tier = df.groupby("Company_Tier")["Offer_Received"].mean().sort_values()
    fig4, ax4 = plt.subplots()
    company_tier.plot(kind="bar", ax=ax4)
    ax4.set_title("Offer Rate by Company Tier")
    st.pyplot(fig4)

    # Salary Analysis
    st.subheader("💰 Salary Distribution")
    fig5, ax5 = plt.subplots()
    df["Salary_LPA"].hist(bins=30, ax=ax5, edgecolor='black')
    ax5.set_title("Salary Distribution (LPA)")
    st.pyplot(fig5)
    st.info("Higher salaries usually correlate with stricter filtering")
    
    # Interview Success Rate
    st.subheader("📊 Interview Success Rate")
    st.metric("Success Rate", f"{interview_success:.2%}")
    
    # Final Insight Box
    st.markdown("---")
    st.success("""
    💡 **Key Takeaways:**
    
    - Hardest roles show lowest offer rates  
    - Internship experience significantly boosts chances  
    - Domain-specific weaknesses exist (AI / GenAI struggle more)  
    - Company tier strongly affects selection difficulty  
    - Salary increases → stricter filtering  
    """)

# =============================
# INSIGHTS
# =============================
elif page == "Insights":
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
    
    ### 🛠️ Suggested Improvements
    - Focus on coding + technical interview preparation  
    - Increase real-world project experience  
    - Improve post-offer engagement to reduce drop-offs 
    """)
 
# ===============================
# INTERACTIVE 'WHAT-IF' SIMULATOR
# ===============================
elif page == "Interactive 'What-If' Simulator":
    st.header("🔮 What-If Simulator")

    improve = st.slider("Improve Offer → Join Conversion (%)", 0, 100, 30)
    
    sim_joined = funnel["Offer"] * (improve / 100)

    st.write("### Simulated Joined Candidates")
    st.metric("Joined", int(sim_joined))
    
    base_rate = funnel["Joined"] / max(funnel["Offer"], 1) * 100
    if improve <= base_rate:
        st.info(f"""If Offer → Join conversion is {improve}%, joined candidates decrease from {funnel['Joined']} → {int(sim_joined)}.""")
    else:
        st.info(f"""If Offer → Join improves to {improve}%, joined candidates increase from {funnel['Joined']} → {int(sim_joined)}.""")
    

# -----------------------------------------------------------------------------------------------------------------------------------
