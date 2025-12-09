# Forest Cover Type Prediction

## 🌲 Project Overview
This project focuses on predicting the forest cover type (the predominant kind of tree cover) for a given 30m x 30m patch of land in the Roosevelt National Forest of northern Colorado. By analyzing cartographic variables such as elevation, soil type, and distance to water, the system classifies the forest cover into one of seven distinct categories.

The solution involves a comprehensive machine learning pipeline including Exploratory Data Analysis (EDA), advanced feature engineering, model benchmarking, and a deployment-ready Streamlit web application.

---

## 📊 Dataset Info
The dataset contains cartographic variables derived from the US Geological Survey (USGS) and US Forest Service (USFS) data.

**Target Variable (Cover_Type):**
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

**Key Features:**
- **Continuous:** Elevation, Aspect, Slope, Horizontal/Vertical Distance to Hydrology, Distance to Roadways & Fire Points, Hillshade indices (9am, Noon, 3pm).
- **Categorical:** 
  - **Wilderness_Area** (4 binary columns): Designations like Rawah, Neota, etc.
  - **Soil_Type** (40 binary columns): Specific soil descriptions.

---

## ⚙️ Methodology
1. **Exploratory Data Analysis (EDA):**
   - Analyzed distributions of continuous variables per cover type.
   - Visualized the relationship between Wilderness Areas and Cover Types.
   - Examined Soil Type frequencies across different classes.
   - Correlation heatmap to identify multicollinearity.

2. **Feature Engineering:**
   - Created interaction features to capture non-linear relationships, such as:
     - `Total_Distance_To_Hydrology` (Euclidean distance)
     - `Elevation_Plus_Vertical_Hydrology` & `Elevation_Minus_Vertical_Hydrology`
     - Interactions between Hydrology, Fire Points, and Roadways.

3. **Preprocessing:**
   - **Splitting:** 80/20 Train-Test split.
   - **Scaling:** Applied `StandardScaler` to continuous features (excluding binary Soil/Wilderness columns).
   - **Mapping:** Adjusted target labels (1-7) to (0-6) for compatibility with certain algorithms (e.g., XGBoost).

4. **Model Benchmarking:**
   - Trained and evaluated 7 algorithms:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Random Forest
     - ExtraTrees Classifier
     - XGBoost

---

## 🛠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deployment:** Streamlit
- **Model Serialization:** Joblib

---

## 📦 Installation Steps

1. **Clone the repository:**

git clone [<repository_url>](https://github.com/aiml-developer/Forest-Cover-Prediction)
cd Forest-Cover-Prediction


2. **Create a virtual environment (optional but recommended):**

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. **Install dependencies:**

pip install pandas numpy seaborn matplotlib scikit-learn xgboost streamlit joblib

---

## 🖥 Usage

### 1. Run the Web App
Launch the Streamlit interface to make real-time predictions.

streamlit run app.py

- Open the provided local URL (usually `http://localhost:8501`).
- Use the sidebar to adjust geographical parameters (Elevation, Aspect, Soil Type, etc.).
- Click **"Predict Forest Cover Type"** to see the result.

### 2. Retrain Models
To regenerate the models and scaler from scratch:
- Open the Jupyter notebook `Forest_Cover_Prediction.ipynb`.
- Run all cells to perform EDA, training, and save the artifacts (`best_forest_model.pkl` and `best_scaler.pkl`).

---

## 🚀 Model Training Steps
The training pipeline is automated within the notebook:
1. **Data Loading:** Reads `train.csv`.
2. **Feature Construction:** Generates 9 new mathematical features based on distances and elevation.
3. **Scaling:** Fits a `StandardScaler` on the training set and saves it.
4. **Evaluation:** Iterates through all 7 models, calculating Accuracy, Precision, Recall, and F1-Score.
5. **Selection:** The model with the highest accuracy is automatically identified.
6. **Serialization:** The best model and scaler are saved to disk for the app to use.

---

## 📈 Results
The project compares multiple classifiers to find the optimal solution.
- **Metrics Used:** Accuracy, Precision (Weighted), Recall (Weighted), F1-Score (Weighted).
- **Winning Model:** The best-performing model (typically Random Forest, ExtraTrees, or XGBoost for this dataset) is saved as `best_forest_model.pkl`.
- **Confusion Matrix:** Generated to visualize misclassifications and understand model behavior.

---

## 📷 Demo
The **Streamlit App** provides an interactive dashboard:
- **Sidebar:** Inputs for Elevation, Slope, Hydrology distances, Wilderness Area selection, and Soil Type.
- **Main Panel:** Displays the predicted class (e.g., "Aspen") along with the confidence/integer ID.

---

## 🔚 Conclusion
This project successfully demonstrates a complete end-to-end Machine Learning workflow. By combining domain-specific feature engineering with robust model benchmarking, we achieved a high-performance classification system. The integration with Streamlit ensures that the complex underlying model is accessible via a simple, user-friendly interface.
