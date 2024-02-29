import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,country: str,year: int,yearly_%_change: int,migrants:int,median_age: int,fertility_rate: int, density(P/Km²): int,urban_pop_%:int,rank:int): # type: ignore

        self.country = country

        self.year = year

        self.yearly_%_change = yearly_%_change

        self.migrants = migrants

        self.median_age = median_age

        self.fertility_rate = fertility_rate

        self.density(P/Km²) = density(P/Km²)
                                      
        self.urban_pop_% = urban_pop_%

        self.rank = rank

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "country": [self.country],
                "year": [self.year],
                "yearly_%_change": [self.yearly_%_change],
                "migrants": [self.migrants],
                "median_age": [self.median_age],
                "fertility_rate": [self.fertility_rate],
                "density(P/Km²)": [self.density(P/Km²)],
                "urban_pop_%": [self.urban_pop_%],
                "rank": [self.rank]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

