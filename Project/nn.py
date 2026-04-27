import knn as knn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class NeuralNetwork(nn.Module):
    def __init__(self,in_features=62,h1=32,h2=16,out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.out(x) 
        return x

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
    #print("Base dataframe") 
    #print(df)


    train = df[df["Type"]== 'Train']
    test = df[df["Type"]== 'Test']   

    train_features = train[["Track ID", "GenreID"]+features]
    test_features = test[["Track ID", "GenreID"]+features] 
    
    training_dataset = Dataset(train_features)
    testing_dataset = Dataset(test_features)
    return training_dataset, testing_dataset

def genreID_toString(genreID):
    dict = {0: "Pop", 1: "Metal", 2: "Disco", 3: "Blues", 4: "Reggae", 5: "Classical", 6: "Rock", 7: "Hiphop", 8: "Country", 9: "Jazz", } 
    return dict[genreID]

if __name__ == "__main__":
    
    # Importing and preparing data:
    using_features = ["zero_cross_rate_mean", "zero_cross_rate_std", "rmse_mean", "rmse_var", "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "spectral_rolloff_mean", "spectral_rolloff_var", "spectral_contrast_mean", "spectral_contrast_var", "spectral_flatness_mean", "spectral_flatness_var", "chroma_stft_1_mean", "chroma_stft_2_mean", "chroma_stft_3_mean", "chroma_stft_4_mean", "chroma_stft_5_mean", "chroma_stft_6_mean", "chroma_stft_7_mean", "chroma_stft_8_mean", "chroma_stft_9_mean", "chroma_stft_10_mean", "chroma_stft_11_mean", "chroma_stft_12_mean", "chroma_stft_1_std", "chroma_stft_2_std", "chroma_stft_3_std", "chroma_stft_4_std", "chroma_stft_5_std", "chroma_stft_6_std", "chroma_stft_7_std", "chroma_stft_8_std", "chroma_stft_9_std", "chroma_stft_10_std", "chroma_stft_11_std", "chroma_stft_12_std", "tempo", "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean", "mfcc_5_mean", "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean", "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean", "mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std", "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std", "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std"]
    datafile = "./data/GenreClassData_30s.txt"
    training_data, testing_data = import_data(datafile,using_features)
    
    
    torch.manual_seed(69)
    model = NeuralNetwork(in_features=len(using_features))
    params = list(model.parameters())
    print(len(params))
    
    # Scaling the data:
    scaler = StandardScaler()
    training_data.features = scaler.fit_transform(training_data.features)
    testing_data.features = scaler.fit_transform(testing_data.features)

    x_train = torch.from_numpy(training_data.features)    
    x_train = x_train.type(torch.float)
    y_train = torch.from_numpy(training_data.genres).type(torch.long) 

    x_test = torch.from_numpy(testing_data.features).type(torch.float)
    y_test = torch.from_numpy(testing_data.genres).type(torch.long) 


    # Training neural network
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 2000
   
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros(epochs)

    for i in range(epochs): 
        output = model(x_train)
        target = y_train

        loss = criterion(output,target)
        
        losses[i] = loss.item()
        
        if i % 50 == 0:
            #print(f'Epoch: {i} and loss: {loss}')
            pass


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    outputs = model(x_test)
    _, predicted = torch.max(outputs,1)
    
    correct = (predicted == y_test).sum().item()
    total = len(y_test)

    cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())
    print(f'CONFUSION MATRIX: {cm}')
    # print(f"---------\n True: {y_test} \n")
    # print(f"---------\n Predicted: {predicted} \n")

    print(f"Correct: {correct} ")
    print(f"Total: {total}")

    accuracy = correct/total
    print(f"Accuracy: {accuracy}")
    #print('Predicted: ', ' '.join(f'{genreID_toString(predicted)}'))

    print(predicted)

    # Sorting results based on genre
    print("Percentage of correct classifications by genre")
    for i in range(10):
        current_genre = genreID_toString(i)
        #print(current_genre)

        mask = y_test == i
        current_preds = predicted[mask]
        num_thisgenre = len(current_preds)
        correct_count = sum(current_preds == i)
        #print(correct_count)
        #print()

        correct_percentage = correct_count/num_thisgenre
        percentage_num = correct_percentage.item()*100

        print(f"{current_genre}: {round(percentage_num,2)}%")

plt.plot(losses)
plt.show()


    