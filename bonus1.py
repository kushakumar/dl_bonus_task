import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

data_path = "incident_calls.csv"
data = pd.read_csv(data_path)

with open('logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)


council_district_options = sorted(data['COUNCIL DISTRICT'].dropna().unique())
council_district_2011_options = sorted(data['Council District 2011'].dropna().unique())
police_district_options = sorted(data['POLICE DISTRICT'].dropna().unique())
neighborhood_options = sorted(data['Neighborhood'].dropna().unique())
city_options = sorted(data['City'].dropna().unique())
state_options = sorted(data['State'].dropna().unique())

st.title("Accident Type Prediction")

council_district = st.selectbox('COUNCIL DISTRICT', council_district_options)
council_district_2011 = st.selectbox('Council District 2011', council_district_2011_options)
police_district = st.selectbox('POLICE DISTRICT', police_district_options)
neighborhood = st.selectbox('Neighborhood', neighborhood_options)
city = st.selectbox('City', city_options)
state = st.selectbox('State', state_options)

input_data = pd.DataFrame([[council_district, council_district_2011, police_district, neighborhood, city, state]],
                          columns=['COUNCIL DISTRICT', 'Council District 2011', 'POLICE DISTRICT', 'Neighborhood', 'City', 'State'])

encoded_df=pd.get_dummies(input_data, columns=['COUNCIL DISTRICT', 'Council District 2011', 'POLICE DISTRICT', 'Neighborhood', 'City', 'State'],
                                  prefix=['COUNCIL DISTRICT', 'Council District 2011', 'POLICE DISTRICT', 'Neighborhood', 'City', 'State'])

cols=['COUNCIL DISTRICT_delaware',	'COUNCIL DISTRICT_ellicott',	'COUNCIL DISTRICT_fillmore',	'COUNCIL DISTRICT_lovejoy',	'COUNCIL DISTRICT_masten',	'COUNCIL DISTRICT_niagara',	'COUNCIL DISTRICT_north',	'COUNCIL DISTRICT_south',	'COUNCIL DISTRICT_university',	'COUNCIL DISTRICT_unknown',	'Council District 2011_delaware',	'Council District 2011_ellicott',	'Council District 2011_fillmore',	'Council District 2011_lovejoy',	'Council District 2011_masten',	'Council District 2011_niagara',	'Council District 2011_north',	'Council District 2011_south',	'Council District 2011_unassigned',	'Council District 2011_university',	'Council District 2011_unknown',	'POLICE DISTRICT_district a',	'POLICE DISTRICT_district b',	'POLICE DISTRICT_district c',	'POLICE DISTRICT_district d',	'POLICE DISTRICT_district e',	'POLICE DISTRICT_unknown',	'Neighborhood_allentown',	'Neighborhood_black rock',	'Neighborhood_broadway fillmore',	'Neighborhood_central',	'Neighborhood_central park',	'Neighborhood_delavan grider',	'Neighborhood_ellicott',	'Neighborhood_elmwood bidwell',	'Neighborhood_elmwood bryant',	'Neighborhood_fillmore-leroy',	'Neighborhood_first ward',	'Neighborhood_fruit belt',	'Neighborhood_genesee-moselle',	'Neighborhood_grant-amherst',	'Neighborhood_hamlin park',	'Neighborhood_hopkins-tifft',	'Neighborhood_kaisertown',	'Neighborhood_kenfield',	'Neighborhood_kensington-bailey',	'Neighborhood_lovejoy',	'Neighborhood_lower west side',	'Neighborhood_masten park',	'Neighborhood_mlk park',	'Neighborhood_north park',	'Neighborhood_parkside',	'Neighborhood_pratt-willert',	'Neighborhood_riverside',	'Neighborhood_schiller park',	'Neighborhood_seneca babcock',	'Neighborhood_seneca-cazenovia',	'Neighborhood_south park',	'Neighborhood_university heights',	'Neighborhood_unknown',	'Neighborhood_upper west side',	'Neighborhood_west hertel',	'Neighborhood_west side',	'City_buffalo',	'State_ny']
for col in cols:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

encoded_df = encoded_df[cols]

if st.button('Predict Accident Type'):
    prediction = model.predict(encoded_df)
    st.write(f"Predicted Accident Type: {prediction[0]}")
