import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. Setup & Config
# ==========================================
st.set_page_config(page_title="Forest Cover Predictor", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌲 Forest Cover Type Prediction System")
st.markdown("Use the sidebar to adjust parameters and predict the forest cover type.")

# ==========================================
# 2. Load Model & Scaler
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_forest_model.pkl')
        scaler = joblib.load('best_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: 'best_forest_model.pkl' or 'best_scaler.pkl' not found. Please run 'python train.py' first.")
        return None, None

model, scaler = load_artifacts()

# ==========================================
# 3. Sidebar Inputs
# ==========================================
st.sidebar.header("🌍 Geographical Features")

def user_input_features():
    # Continuous Variables
    elevation = st.sidebar.slider("Elevation (meters)", 1800, 4000, 2500)
    aspect = st.sidebar.slider("Aspect (degrees azimuth)", 0, 360, 100)
    slope = st.sidebar.slider("Slope (degrees)", 0, 66, 15)
    
    st.sidebar.header("💧 Hydrology & Features")
    h_dist_hydro = st.sidebar.number_input("Horiz. Dist. to Water", 0, 1500, 200)
    v_dist_hydro = st.sidebar.number_input("Vert. Dist. to Water", -200, 600, 50)
    h_dist_road = st.sidebar.number_input("Horiz. Dist. to Roadways", 0, 7000, 1000)
    h_dist_fire = st.sidebar.number_input("Horiz. Dist. to Fire Points", 0, 7000, 1000)
    
    st.sidebar.header("☀️ Sunlight (Hillshade 0-255)")
    hillshade_9am = st.sidebar.slider("Hillshade 9am", 0, 255, 200)
    hillshade_noon = st.sidebar.slider("Hillshade Noon", 0, 255, 200)
    hillshade_3pm = st.sidebar.slider("Hillshade 3pm", 0, 255, 150)
    
    st.sidebar.header("🌲 Area & Soil")
    wilderness_area = st.sidebar.selectbox(
        "Wilderness Area",
        ("Area 1 (Rawah)", "Area 2 (Neota)", "Area 3 (Comanche Peak)", "Area 4 (Cache la Poudre)")
    )
    
    soil_type = st.sidebar.selectbox(
        "Soil Type (1-40)",
        list(range(1, 41))
    )
    
    data = {
        'Elevation': elevation,
        'Aspect': aspect,
        'Slope': slope,
        'Horizontal_Distance_To_Hydrology': h_dist_hydro,
        'Vertical_Distance_To_Hydrology': v_dist_hydro,
        'Horizontal_Distance_To_Roadways': h_dist_road,
        'Horizontal_Distance_To_Fire_Points': h_dist_fire,
        'Hillshade_9am': hillshade_9am,
        'Hillshade_Noon': hillshade_noon,
        'Hillshade_3pm': hillshade_3pm,
        'Wilderness_Area': wilderness_area,
        'Soil_Type': soil_type
    }
    return data

input_data = user_input_features()

# ==========================================
# 4. Preprocessing Function (UPDATED)
# ==========================================
def preprocess_input(data, scaler):
    # 1. Create Data Dictionary with Base Features
    # Order doesn't matter here yet, we are just calculating values
    base = {
        'Elevation': data['Elevation'],
        'Aspect': data['Aspect'],
        'Slope': data['Slope'],
        'Horizontal_Distance_To_Hydrology': data['Horizontal_Distance_To_Hydrology'],
        'Vertical_Distance_To_Hydrology': data['Vertical_Distance_To_Hydrology'],
        'Horizontal_Distance_To_Roadways': data['Horizontal_Distance_To_Roadways'],
        'Horizontal_Distance_To_Fire_Points': data['Horizontal_Distance_To_Fire_Points'],
        'Hillshade_9am': data['Hillshade_9am'],
        'Hillshade_Noon': data['Hillshade_Noon'],
        'Hillshade_3pm': data['Hillshade_3pm']
    }

    # 2. Feature Engineering (Calculations)
    base['Total_Distance_To_Hydrology'] = (base['Vertical_Distance_To_Hydrology']**2 + base['Horizontal_Distance_To_Hydrology']**2)**0.5
    base['Elevation_Plus_Vertical_Hydrology'] = base['Elevation'] + base['Vertical_Distance_To_Hydrology']
    base['Elevation_Minus_Vertical_Hydrology'] = base['Elevation'] - base['Vertical_Distance_To_Hydrology']
    base['Hydrology_Plus_Fire_Points'] = base['Horizontal_Distance_To_Hydrology'] + base['Horizontal_Distance_To_Fire_Points']
    base['Hydrology_Minus_Fire_Points'] = base['Horizontal_Distance_To_Hydrology'] - base['Horizontal_Distance_To_Fire_Points']
    base['Hydrology_Plus_Roadways'] = base['Horizontal_Distance_To_Hydrology'] + base['Horizontal_Distance_To_Roadways']
    base['Hydrology_Minus_Roadways'] = base['Horizontal_Distance_To_Hydrology'] - base['Horizontal_Distance_To_Roadways']
    base['Fire_Points_Plus_Roadways'] = base['Horizontal_Distance_To_Fire_Points'] + base['Horizontal_Distance_To_Roadways']
    base['Fire_Points_Minus_Roadways'] = base['Horizontal_Distance_To_Fire_Points'] - base['Horizontal_Distance_To_Roadways']

    # 3. Create List for Scaling (Strict Order Matching Training)
    # This list MUST match the columns used in 'continuous_cols' during training
    cols_to_scale_values = [
        base['Elevation'], base['Aspect'], base['Slope'], 
        base['Horizontal_Distance_To_Hydrology'], base['Vertical_Distance_To_Hydrology'], 
        base['Horizontal_Distance_To_Roadways'], base['Horizontal_Distance_To_Fire_Points'], 
        base['Hillshade_9am'], base['Hillshade_Noon'], base['Hillshade_3pm'],
        base['Total_Distance_To_Hydrology'], base['Elevation_Plus_Vertical_Hydrology'], base['Elevation_Minus_Vertical_Hydrology'],
        base['Hydrology_Plus_Fire_Points'], base['Hydrology_Minus_Fire_Points'], base['Hydrology_Plus_Roadways'],
        base['Hydrology_Minus_Roadways'], base['Fire_Points_Plus_Roadways'], base['Fire_Points_Minus_Roadways']
    ]

    # 4. Scale Values directly (Bypassing Feature Name Check)
    # Reshape to (1, -1) because it's a single sample
    scaled_values = scaler.transform(np.array(cols_to_scale_values).reshape(1, -1))[0]

    # 5. One-Hot Encoding (Manual)
    # Wilderness Areas (1-4)
    wa_map = {
        "Area 1 (Rawah)": 1,
        "Area 2 (Neota)": 2,
        "Area 3 (Comanche Peak)": 3,
        "Area 4 (Cache la Poudre)": 4
    }
    selected_wa = wa_map[data['Wilderness_Area']]
    wa_values = [1 if i == selected_wa else 0 for i in range(1, 5)]

    # Soil Types (1-40)
    selected_soil = data['Soil_Type']
    soil_values = [1 if i == selected_soil else 0 for i in range(1, 41)]

    # 6. Combine All Features into Final Array
    # Structure: [Scaled Continuous] + [Wilderness 1-4] + [Soil 1-40]
    final_features = np.concatenate([scaled_values, wa_values, soil_values])
    
    # Return as 2D array for model
    return final_features.reshape(1, -1)

# ==========================================
# 5. Main Display Area
# ==========================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Current Input Parameters")
    st.write(f"**Wilderness Area:** {input_data['Wilderness_Area']}")
    st.write(f"**Elevation:** {input_data['Elevation']} m")
with col2:
    st.write("") # Spacer
    st.write("") 
    st.write(f"**Soil Type:** Type {input_data['Soil_Type']}")
    st.write(f"**Slope:** {input_data['Slope']}°")

if st.button("Predict Forest Cover Type"):
    if model and scaler:
        try:
            # Process inputs
            processed_df = preprocess_input(input_data, scaler)
            
            # Predict
            prediction_idx = model.predict(processed_df)[0]
            
            # Mapping logic (0-6 -> 1-7)
            predicted_class = prediction_idx + 1 
            
            cover_type_names = {
                1: "Spruce/Fir",
                2: "Lodgepole Pine",
                3: "Ponderosa Pine",
                4: "Cottonwood/Willow",
                5: "Aspen",
                6: "Douglas-fir",
                7: "Krummholz"
            }
            
            result_name = cover_type_names.get(predicted_class, "Unknown")
            
            st.success(f"🌲 Predicted Cover Type: {predicted_class} - {result_name}")
            
            # Probability Chart
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(processed_df)[0]
                prob_df = pd.DataFrame({
                    "Cover Type": list(cover_type_names.values()),
                    "Probability": probs
                })
                st.bar_chart(prob_df.set_index("Cover Type"))
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.warning("Ensure 'train.py' and 'app.py' use the EXACT same feature engineering and column order.")
    else:
        st.warning("Model not loaded.")
