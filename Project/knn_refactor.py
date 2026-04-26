import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, dataframe):
        self.raw_data = dataframe.to_numpy()
        self.ids = self.raw_data[:,0]
        self.genres = self.raw_data[:,1].astype(int)
        self.features = self.raw_data[:,2:]

    def get_genre(self,genreid):
        mask = self.genres == genreid
        return self.features[mask]
    
def import_data(filename: str, features: list[str]) -> tuple[Dataset,Dataset]:
    df = pd.read_csv(filename,sep="\t")
    print("Base dataframe") 
    print(df)


    train = df[df["Type"]== 'Train']
    test = df[df["Type"]== 'Test']   

    train_features = train[["Track ID", "GenreID"]+features]
    test_features = test[["Track ID", "GenreID"]+features] 
    
    training_dataset = Dataset(train_features)
    testing_dataset = Dataset(test_features)
    return training_dataset, testing_dataset

def knn():
    pass

if __name__ == "__main__":
    using_features = ["spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"] 
    
    datafile = "./data/GenreClassData_30s.txt"
    training_data, testing_data = import_data(datafile,using_features)
    print(training_data.features)