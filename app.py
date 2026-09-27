import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Water Pipe Leak Detection",
    page_icon="💧",
    layout="centered"
)


# =========================================================
# LOAD SAVED MODEL AND FEATURES
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("water_leak_random_forest.pkl")
    features = joblib.load("water_leak_features.pkl")

    return model, features


model, features = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("💧 Water Pipe Leak Detection")

st.write(
    "Enter the water pipe and flow information below "
    "to predict whether a leak is present."
)


# =========================================================
# INPUT FEATURES
# =========================================================

st.subheader("Pipe Information")


timestep = st.number_input(
    "Timestep",
    min_value=0,
    value=6000,
    step=1
)


diameter = st.number_input(
    "Diameter (Inch)",
    min_value=0.0,
    value=12.0,
    step=0.1
)


roughness = st.number_input(
    "Roughness (mm)",
    min_value=0.0,
    value=0.15,
    step=0.01
)


pressure = st.number_input(
    "Pressure (PSI)",
    min_value=0.0,
    value=55.0,
    step=0.1
)


flow = st.number_input(
    "Flow Rate (LPM)",
    min_value=0.0,
    value=120.0,
    step=0.1
)


material = st.selectbox(
    "Pipe Material",
    ["HDPE", "PVC"]
)


# =========================================================
# ONE-HOT ENCODING FOR PIPE MATERIAL
# =========================================================

if material == "HDPE":

    material_hdpe = 1
    material_pvc = 0

else:

    material_hdpe = 0
    material_pvc = 1


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Leak", use_container_width=True):

    # Create input DataFrame
    sample = pd.DataFrame([{

        "Timestep": timestep,

        "Diameter_Inch": diameter,

        "Roughness_mm": roughness,

        "Pressure_PSI": pressure,

        "Flow_LPM": flow,

        "Material_HDPE": material_hdpe,

        "Material_PVC": material_pvc

    }])


    # =====================================================
    # ENSURE CORRECT FEATURE ORDER
    # =====================================================

    sample = sample[features]


    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    prediction = model.predict(sample)[0]


    # Get prediction probabilities
    probabilities = model.predict_proba(sample)[0]

    no_leak_probability = probabilities[0]

    leak_probability = probabilities[1]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error("🚨 LEAK DETECTED")

    else:

        st.success("✅ NO LEAK DETECTED")


    # =====================================================
    # DISPLAY PROBABILITIES
    # =====================================================

    st.write(
        f"**No Leak Probability:** {no_leak_probability:.2%}"
    )

    st.write(
        f"**Leak Probability:** {leak_probability:.2%}"
    )


    # Progress bar for leak probability
    st.progress(float(leak_probability))


    # =====================================================
    # DISPLAY INPUT DATA
    # =====================================================

    with st.expander("View Input Data"):

        st.dataframe(sample)