import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

from pub_recommender import PubRecommender

columns = [
        "fsa_id",
        "name",
        "address",
        "postcode",
        "easting",
        "northing",
        "latitude",
        "longitude",
        "local_authority",
    ]
    
df = pd.read_csv(
    "./data/open_pubs.csv",
    names=columns,
    index_col=0,
)

# Clean data
df.latitude.replace("\\N", np.nan, inplace=True)
df.longitude.replace("\\N", np.nan, inplace=True)
df.dropna(inplace=True)
print("Dropped Null values")

df.reset_index(inplace=True)

# Scale Latitude and Longitude features
scaler = StandardScaler()
tnf_df = scaler.fit_transform(df.iloc[:, [5, 6]].values)

# Nearest Neighbors model
model = NearestNeighbors(metric="euclidean", n_jobs=-1)
model.fit(tnf_df)

recommender = PubRecommender(scaler, model, df)

# saving the model
joblib.dump(recommender, "model.obj")
print("Model Saved")
