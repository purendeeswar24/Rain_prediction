import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
try:
    model = joblib.load("rainxgb.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model = None  # Ensure model is None if loading fails

# CSS styling
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        color: #FF5733;
        text-align: center;
        font-weight: bold;
    }
    .header {
        font-size: 24px;
        color: #5C6BC0;
        font-weight: bold;
        text-align: center;
    }
    .input-label {
        font-size: 18px;
        color: #3E2723;
        font-weight: bold;
    }
    .input-field {
        margin-bottom: 20px;
        padding: 10px;
        border: 2px solid #FFB300;
        border-radius: 5px;
        background-color: #FFF9C4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the app
st.markdown('<div class="title">Rain Prediction App</div>', unsafe_allow_html=True)
st.write("Use this app to predict if it will rain based on the input weather conditions.")  # Usage instruction

# Input fields for user data
st.markdown('<div class="header">Input Weather Conditions</div>', unsafe_allow_html=True)
st.image("rain_pred.jpg", caption="Rain Prediction Illustration", use_column_width=True)

# Define mappings for categorical features
conditions_mapping = { 
    "Smoke": 0, "Mist": 1, "Clear": 2, "Widespread Dust": 3, "Fog": 4, 
    "Scattered Clouds": 5, "Partly Cloudy": 6, "Shallow Fog": 7, 
    "Mostly Cloudy": 8, "Light Rain": 9, "Partial Fog": 10, "Patches of Fog": 11, 
    "Thunderstorms and Rain": 12, "Heavy Fog": 13, "Light Drizzle": 14, 
    "Rain": 15, "Unknown": 16, "Blowing Sand": 17, "Overcast": 18, 
    "Thunderstorm": 19, "Light Thunderstorms and Rain": 20, "Drizzle": 21, 
    "Light Fog": 22, "Light Thunderstorm": 23, "Heavy Rain": 24, 
    "Heavy Thunderstorms and Rain": 25, "Thunderstorms with Hail": 26, 
    "Squalls": 27, "Light Sandstorm": 28, "Light Rain Showers": 29, 
    "Volcanic Ash": 30, "Light Haze": 31, "Sandstorm": 32, "Funnel Cloud": 33, 
    "Rain Showers": 34, "Heavy Thunderstorms with Hail": 35, 
    "Light Hail Showers": 36, "Light Freezing Rain": 37
}
wind_direction_mapping = {
    "North": 0, "West": 1, "WNW": 2, "East": 3, "NW": 4, "WSW": 5, "ESE": 6, 
    "ENE": 7, "SE": 8, "SW": 9, "NNW": 10, "NE": 11, "SSE": 12, "NNE": 13, 
    "SSW": 14, "South": 15, "Variable": 16
}

# Create a two-column layout for input fields
col1, col2 = st.columns(2)

# Dropdown selection for 'conditions' and map to numeric
with col1:
    conditions = st.selectbox("Conditions", options=list(conditions_mapping.keys()), key="conditions")
    conditions_encoded = conditions_mapping[conditions]

    # Input for 'dew_temp'
    dew_temp = st.number_input("Dew Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)

    # Drop-downs for binary values
    fog = st.selectbox("Fog", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="fog")
    balls_of_ice = st.selectbox("Balls of Ice", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="balls_of_ice")
    snow = st.selectbox("Snow", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="snow")
    tornado = st.selectbox("Tornado", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="tornado")

with col2:
    # Number input for 'humidity', 'atm_pressure', and 'temp'
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
    atm_pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900, max_value=1100, value=1013)
    temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)

    # Dropdown for 'wind_direction' and map to numeric
    wind_direction = st.selectbox("Wind Direction", options=list(wind_direction_mapping.keys()), key="wind_direction")
    wind_direction_encoded = wind_direction_mapping[wind_direction]

# Button to make prediction
if st.button("Predict"):
    if model is not None:
        # Validate inputs
        if dew_temp < -50 or dew_temp > 50:
            st.error("Dew Temperature must be between -50°C and 50°C.")
        elif humidity < 0 or humidity > 100:
            st.error("Humidity must be between 0% and 100%.")
        elif atm_pressure < 900 or atm_pressure > 1100:
            st.error("Atmospheric Pressure must be between 900 hPa and 1100 hPa.")
        elif temp < -50 or temp > 50:
            st.error("Temperature must be between -50°C and 50°C.")
        else:
            # Prepare input data as a 2D array
            input_data = np.array([[conditions_encoded, dew_temp, fog, balls_of_ice, humidity, 
                                    atm_pressure, snow, temp, tornado, wind_direction_encoded]])

            # Make prediction
            try:
                prediction = model.predict(input_data)

                # Interpret prediction
                if prediction[0] == 1:
                    st.success("Rain is predicted to come!")
                else:
                    st.success("No rain is predicted.")

                # Generate a bar plot of input features for visualization
                feature_labels = ['Conditions', 'Dew Temp', 'Humidity', 'Pressure', 'Temperature']
                feature_values = [conditions_encoded, dew_temp, humidity, atm_pressure, temp]

                plt.figure(figsize=(10, 5))
                sns.barplot(x=feature_labels, y=feature_values, palette="viridis")
                plt.title("Input Weather Conditions")
                plt.ylabel("Values")
                plt.xticks(rotation=45)
                st.pyplot(plt)  # Display plot in Streamlit

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Model not loaded. Cannot make predictions.")
