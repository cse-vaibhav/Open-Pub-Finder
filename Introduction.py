import streamlit as st
import joblib

model = joblib.load("model.obj")

st.title("Hello World!")

st.write("# Dataset")
st.write(" We are using open source map data provided by https://www.getthedata.com/open-pubs")

st.write(model.data.head())

st.write(f"The dataset has {model.data.shape[0]} rows and {model.data.shape[1]} features")

st.write("## Features")
st.write("We have following features in our dataset")

st.write("""

* __fsa_id__ : Food Standard Agency's ID for this pub.
* __name__ : Name of the pub
* __address__ : Address fields separated by comma
* __postcode__ : Postcode of the pub
* __easting__ : Integer column
* __northing__ : Integer column
* __latitude__ : Latitude co-ordinate of the pub
* __longitude__ : Longitude co-ordinate of the pub
""")

st.write("## Dataset Description")
model.data["longitude"] = model.data["longitude"].astype("float64")
model.data["latitude"] = model.data["latitude"].astype("float64")

st.write(model.data.iloc[:, 1:].describe())

