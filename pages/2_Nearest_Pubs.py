import streamlit as st

from pub_recommender import PubRecommender
import joblib
import pandas as pd

model: PubRecommender = joblib.load("model.obj")
k = 5 # Number of recommendations to return
st.title("Open Pub Recommendation")

fields = {
    "latitude": st.number_input("Latitude"),
    "longitude": st.number_input("Longitude")
}

if st.button("Search Pubs"):

    data = list(fields.values())
    recommendations = model.get_k_recommendations([data], k)

    recommendation_text = ""
    for recommendation in recommendations.values:
        s = pd.Series(recommendation, index=recommendations.columns)

        recommendation_text += "## " + s["name"]
        recommendation_text += " - " + s["local_authority"]
        recommendation_text += "\n#### " + s["address"]
        recommendation_text += "\n##### " + "Postcode: " + s["postcode"]
        recommendation_text += "\n> " + f"### Distance: {s['distance']:.2f} miles"
        recommendation_text += "\n"

    st.write(recommendation_text, unsafe_allow_html=True)

