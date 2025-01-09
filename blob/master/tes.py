pip install streamlit
pip install scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title("Prediksi Harga Tawar Properti di Malaysia")
st.write("Aplikasi ini membantu memprediksi harga properti berdasarkan parameter yang Anda masukkan.")

# Load and preprocess data
@st.cache_data
def load_data():
    dataset = pd.read_csv('malaysia_property_for_sale.csv')
    
    # Data cleaning
    dataset['list_price'] = dataset['list_price'].str.replace('RM', '').str.replace(',', '').astype(float)
    dataset['unit_price'] = dataset['unit_price'].str.replace('RM', '').str.replace(',', '').str.replace('/ m2', '').str.replace('(', '').str.replace(')', '').astype(float)
    dataset['area'] = dataset['area'].str.replace(' m2', '').str.replace(',', '').astype(float)
    
    # Encoding
    dataset['location_encoded'] = dataset['location'].astype('category').cat.codes
    dataset['type_encoded'] = dataset['type'].astype('category').cat.codes
    
    return dataset

data = load_data()

# Display dataset preview
if st.checkbox("Tampilkan dataset", False):
    st.write(data.head())

# Define inputs
st.sidebar.header("Input Parameter")
number_bedroom = st.sidebar.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
number_bathroom = st.sidebar.number_input("Jumlah Kamar Mandi", min_value=0, value=2)
location = st.sidebar.selectbox("Lokasi", data['location'].unique())
area_m2 = st.sidebar.number_input("Area (m2)", min_value=0, value=100)
type_property = st.sidebar.selectbox("Tipe Properti", data['type'].unique())
unit_price_rm_m2 = st.sidebar.number_input("Harga per m2 (RM)", min_value=0, value=5000)

# Map inputs to encoded values
location_encoded = data.loc[data['location'] == location, 'location_encoded'].iloc[0]
type_encoded = data.loc[data['type'] == type_property, 'type_encoded'].iloc[0]

# Prepare model data
X = data[['number_bedroom', 'number_bathroom', 'location_encoded', 'area', 'type_encoded', 'unit_price']].values
y = data['list_price'].values

# Handle missing or infinite values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model using Ridge Regression for stability
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predict on both training and test data
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)

# Calculate evaluation metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Display evaluation results
# st.subheader("Akurasi Model")
# st.write(f"**R-squared (R²) pada Data Training:** {train_r2:.2f}")
# st.write(f"**R-squared (R²) pada Data Testing:** {test_r2:.2f}")
# st.write(f"**Mean Squared Error (MSE) pada Data Training:** {train_mse:,.2f}")
# st.write(f"**Mean Squared Error (MSE) pada Data Testing:** {test_mse:,.2f}")

# Prediction
user_input = np.array([
    number_bedroom, number_bathroom, location_encoded, area_m2, type_encoded, unit_price_rm_m2
]).reshape(1, -1)
user_input_scaled = scaler.transform(user_input)

estimated_price = ridge.predict(user_input_scaled)

# Display result
st.subheader("Estimasi Harga")
st.write(f"Harga properti yang diperkirakan adalah RM {estimated_price[0]:,.2f}")
