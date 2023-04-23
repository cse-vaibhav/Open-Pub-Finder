import streamlit as st
import joblib

from pandas import Series
from pub_recommender import PubRecommender
model: PubRecommender = joblib.load("model.obj")

st.title("Map for Pubs")

# Store the initial value of widgets in session state
if "pub_name" not in st.session_state:
    st.session_state.pub_name = model.data.name[0]


def get_loc():
    pub_name = st.session_state.pub_name
    loc = model.data.copy()

    loc["longitude"] = loc["longitude"].astype("float64")
    loc["latitude"] = loc["latitude"].astype("float64")

    if not pub_name:
        st.write("Select")
    
    loc = loc[loc.name == pub_name]
    st.map(loc)

fields = {
    "Name": st.selectbox(
        "Pub Name", 
        options=model.data.name.unique(), 
        on_change=get_loc,
        key="pub_name"
    )
}


