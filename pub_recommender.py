from pandas import DataFrame
from numpy import ndarray

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from typing import List


class PubRecommender:
    def __init__(self, scaler, model, data):
        self.scaler: StandardScaler = scaler
        self.model: NearestNeighbors = model
        self.data: DataFrame = data

    def get_k_recommendations(self, user_data: ndarray, k: int) -> DataFrame:
        user_data = self.scaler.transform(user_data)
        distances, indices = self.model.kneighbors(
            user_data, n_neighbors=k
        )
        indices = indices[0].tolist()
        recommendations = self.data.iloc[indices, :].copy()
        recommendations["distance"] = distances.reshape(-1, 1)
        return recommendations
