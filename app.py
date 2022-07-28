import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

model = joblib.load('Models/lgbm_tuned_final.joblib')

st.set_page_config(page_title = "Site Energy Consumption App",
                   page_icon = "electric", layout = "wide")

# Creating option list for dropdown menu

option_facity_type = ['Grocery_store_or_food_market',
       'Warehouse_Distribution_or_Shipping_center',
       'Retail_Enclosed_mall', 'Education_Other_classroom',
       'Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage',
       'Office_Uncategorized', 'Data_Center', 'Commercial_Other',
       'Mixed_Use_Predominantly_Commercial',
       'Office_Medical_non_diagnostic', 'Education_College_or_university',
       'Industrial', 'Laboratory',
       'Public_Assembly_Entertainment_culture',
       'Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized',
       'Lodging_Hotel', 'Retail_Strip_shopping_mall',
       'Education_Uncategorized', 'Health_Care_Inpatient',
       'Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting',
       'Religious_worship', 'Mixed_Use_Commercial_and_Residential',
       'Office_Bank_or_other_financial', 'Parking_Garage',
       'Commercial_Unknown', 'Service_Vehicle_service_repair_shop',
       'Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation',
       'Service_Uncategorized', 'Warehouse_Refrigerated',
       'Food_Service_Uncategorized', 'Health_Care_Uncategorized',
       'Food_Service_Other', 'Public_Assembly_Movie_Theater',
       'Food_Service_Restaurant_or_cafeteria', 'Food_Sales',
       'Public_Assembly_Uncategorized', 'Nursing_Home',
       'Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare',
       '5plus_Unit_Building', 'Multifamily_Uncategorized',
       'Lodging_Dormitory_or_fraternity_sorority',
       'Public_Assembly_Library', 'Public_Safety_Uncategorized',
       'Public_Safety_Fire_or_police_station', 'Office_Mixed_use',
       'Public_Assembly_Other', 'Public_Safety_Penitentiary',
       'Health_Care_Outpatient_Uncategorized', 'Lodging_Other',
       'Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse',
       'Public_Assembly_Stadium', 'Lodging_Uncategorized',
       '2to4_Unit_Building', 'Warehouse_Uncategorized']

option_building_class = ['Commercial', 'Residential']

option_state_factor = ['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10', 'State_11']

features = ['floor_area', 'energy_star_rating', 'facility_type', 'ELEVATION', 'State_Factor', 'building_class_Residential', 'snowfall_inches']


st.markdown("<h1 style = 'text-align: center;'> Site Energy Consumption App </h1>", unsafe_allow_html = True)

def main(): 
    with st.form('prediction_form'): 

        st.subheader("Enter below details to calculate Site Energy Consumption: ")

        floor_area = st.number_input("Floor Area (Sq. FT): ", min_value = 100, max_value = 999999999)
        energy_star_rating = st.slider('Energy Star Rating: ', min_value = 0, max_value = 100)
        facility_type = st.selectbox('Facility Type: ', options = option_facity_type)
        ELEVATION = st.number_input('Elevation: ', min_value = 0, max_value = 2000)
        State_Factor = st.selectbox('State: ', options = option_state_factor)
        building_class_Residential = st.selectbox('Building Class: ', options = option_building_class)
        snowfall_inches = st.number_input('Snowfall (inches): ', min_value = 0, max_value = 200)

        submit = st.form_submit_button("Predict Result")


    if submit: 
        facility_type = pd.Series(facility_type).astype('category').cat.codes
        State_Factor = pd.Series(State_Factor).astype('category').cat.codes
        if building_class_Residential == 'Residential': 
            building_class_Residential = 1
        else: 
            building_class_Residential = 0

        
        data = np.array([floor_area, energy_star_rating, facility_type, ELEVATION, 
                         State_Factor, building_class_Residential, snowfall_inches]).reshape(1,-1)

        pred = model.predict(data)

        st.write(f"The predicted Energy Consumption is:  {pred[0]}")

if __name__ == '__main__':
    main()
