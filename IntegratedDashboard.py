import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_survey_data.csv')
        return df
    except FileNotFoundError:
        st.error("Cleaned data file not found. Please run data_preprocessing.py first.")
        st.stop()

df = load_data()

st.title("Customized Mental Health Dashboard")
st.write("This dashboard provides simple visualizations for key survey questions and a predictor for mental health risk level.")

# Define a function to calculate a numerical risk score
def calculate_risk_score(row):
    score = 0
    # mental_health_impact: higher score for negative impact
    if row['mental_health_impact'] == 'Very negative': score += 4
    elif row['mental_health_impact'] == 'Somewhat negative': score += 3
    elif row['mental_health_impact'] == 'Neutral': score += 2
    elif row['mental_health_impact'] == 'Somewhat positive': score += 1
    
    # anxiety_stress
    if row['anxiety_stress'] == 'Frequently': score += 3
    elif row['anxiety_stress'] == 'Occasionally': score += 2
    elif row['anxiety_stress'] == 'Rarely': score += 1

    # loneliness
    if row['loneliness'] == 'Always': score += 3
    elif row['loneliness'] == 'Often': score += 2
    elif row['loneliness'] == 'Sometimes': score += 1
    
    # depression_sadness_impact
    if row['depression_sadness_impact'] == 'Yes, significantly': score += 3
    elif row['depression_sadness_impact'] == 'Yes, somewhat': score += 2

    return score

df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# --- Simple & Understandable Visualizations for Important Questions ---
st.header("1. Key Survey Question Visualizations")

def create_bar_chart(data, column, title):
    counts = data[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.bar(counts, x=column, y='count', title=title)
    fig.update_layout(xaxis_title=column.replace('_', ' ').title(), yaxis_title="Number of Respondents")
    st.plotly_chart(fig)

important_questions_viz = [
    'social_media_usage',
    'primary_reason',
    'mental_health_impact',
    'anxiety_stress',
    'loneliness',
    'productivity_impact',
    'sleep_quality_impact',
    'comparison_frequency',
    'break_frequency',
    'depression_sadness_impact'
]

for q in important_questions_viz:
    create_bar_chart(df, q, f"Distribution of {q.replace('_', ' ').title()}")

st.subheader("Distribution of Calculated Risk Scores")
fig_risk_score_dist = px.histogram(df, x='risk_score', nbins=20, title="Risk Score Distribution")
st.plotly_chart(fig_risk_score_dist)


# --- Proper Risk Level Predictor (Moved to Sidebar) ---
with st.sidebar.expander("Mental Health Risk Predictor", expanded=True):
    st.header("Mental Health Risk Level Predictor")
    st.write("""
        This predictor estimates a numerical mental health risk score based on your responses to important questions.
        A higher score indicates a greater potential negative impact on mental health.
    """)

    # Prepare data for the regressor model
    predictor_features = [
        'Age',
        'social_media_usage',
        'anxiety_stress',
        'loneliness',
        'productivity_impact',
        'sleep_quality_impact',
        'comparison_frequency',
        'break_frequency',
        'depression_sadness_impact'
    ]
    target_risk_score = 'risk_score'

    X_reg = df[predictor_features]
    y_reg = df[target_risk_score]

    # One-Hot Encode categorical features
    X_reg_encoded = pd.get_dummies(X_reg, columns=X_reg.select_dtypes(include='object').columns)

    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_encoded, y_reg, test_size=0.25, random_state=42)

    # Train the RandomForestRegressor model
    from sklearn.ensemble import RandomForestRegressor
    regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor_model.fit(X_train_reg, y_train_reg)

    # User input for risk score prediction
    st.subheader("Get Your Predicted Mental Health Risk Score")

    age_input = st.slider("Age", 10, 80, 25, key='age_risk')
    social_media_usage_input = st.selectbox("How often do you use social media?", df['social_media_usage'].unique(), key='sm_usage_risk')
    anxiety_stress_input = st.selectbox("Do you ever feel anxious or stressed due to social media use?", df['anxiety_stress'].unique(), key='anxiety_risk')
    loneliness_input = st.selectbox("Do you experience feelings of loneliness after using social media?", df['loneliness'].unique(), key='loneliness_risk')
    productivity_input = st.selectbox("Do you feel that social media has affected your productivity in daily life?", df['productivity_impact'].unique(), key='productivity_risk')
    sleep_quality_input = st.selectbox("Do you think social media usage affects your sleep quality?", df['sleep_quality_impact'].unique(), key='sleep_risk')
    comparison_input = st.selectbox("How often do you compare yourself to others on social media?", df['comparison_frequency'].unique(), key='comparison_risk')
    break_frequency_input = st.selectbox("How often do you feel the need to take a break from social media?", df['break_frequency'].unique(), key='break_risk')
    depression_sadness_input = st.selectbox("Has social media contributed to feelings of depression or sadness in your life?", df['depression_sadness_impact'].unique(), key='depression_risk')

    if st.button("Predict My Risk Level"):
        user_risk_data = pd.DataFrame({
            'Age': [age_input],
            'social_media_usage': [social_media_usage_input],
            'anxiety_stress': [anxiety_stress_input],
            'loneliness': [loneliness_input],
            'productivity_impact': [productivity_input],
            'sleep_quality_impact': [sleep_quality_input],
            'comparison_frequency': [comparison_input],
            'break_frequency': [break_frequency_input],
            'depression_sadness_impact': [depression_sadness_input]
        })

        user_risk_data_encoded = pd.get_dummies(user_risk_data)
        user_risk_data_aligned = user_risk_data_encoded.reindex(columns=X_reg_encoded.columns, fill_value=0)

        predicted_risk_score = regressor_model.predict(user_risk_data_aligned)

        st.subheader("Predicted Mental Health Risk Level:")

        # Define risk thresholds and colors
        score = predicted_risk_score[0]
        if score <= 4:
            risk_level = "Low Risk"
            color = "green"
        elif score <= 8:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"

        st.markdown(f"<h3 style='color:{color};'>{risk_level} (Score: {score:.2f})</h3>", unsafe_allow_html=True)
        st.info("A higher score indicates a greater potential negative impact on mental health.")
