import streamlit as st
from chains import AppFeatureChain

st.set_page_config(page_title="AI-Powered App Feature Suggester", layout="centered")
st.title("🤖 App Feature Suggester using Gemini AI")

# Initialize the chain
chain = AppFeatureChain()

# Step 1: App Description Input
app_description = st.text_area(
    "🔍 Describe your App:",
    placeholder="E.g. A health tracking app with location and photo features"
)

if st.button("Analyze App Description"):
    suggested_features = chain.fetch_app_features(app_description)
    st.session_state['suggested_features'] = suggested_features
    st.session_state['feature_flags'] = {}  # reset previous choices

# Step 2: Feature Selection Grid
if 'suggested_features' in st.session_state:
    st.subheader("✅ Suggested Features Based on App Description")
    st.write(st.session_state['suggested_features'])

    st.subheader("📌 Select Features You Will Use")

    feature_flags = {}
    cols = st.columns(2)

    for i, feature in enumerate(st.session_state['suggested_features']):
        with cols[i % 2]:
            choice = st.radio(
                f"🔧 {feature}",
                ["Yes", "No"],
                key=f"{feature}_choice",
                horizontal=True
            )
            feature_flags[feature] = choice

    # Store choices in session
    st.session_state['feature_flags'] = feature_flags

    # Step 3: Generate Markdown Table
    if st.button("Generate Feature Summary"):
        table = chain.generate_feature_table(st.session_state['feature_flags'])
        st.markdown("### 📝 Feature Summary")
        st.markdown(table)
