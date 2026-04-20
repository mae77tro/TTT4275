import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def euclidean_distance(x,ref):
    return np.matmul(np.transpose(x - ref["mu"]),(x - ref["mu"]))

def mahalanobis_distance(x,ref):
    d = x-ref["mu"]
    return d @ np.linalg.solve(ref["sigma"], d)

def import_data(filename):
    df = pd.read_csv(filename,sep="\t")
    train = df[df["Type"]== 'Train']
    test = df[df["Type"]== 'Test']   
    #print(train)
    #print(test)

    train_features = train[["Track ID", "GenreID", "spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"]]
    test_features = test[["Track ID", "GenreID", "spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"]] 
    return train_features, test_features

#def find_mahalanobis_distance():
#    return
#
# 

def divide_by_class(df):
    pop = df[df["GenreID"]== '0']
    metal = df[df["GenreID"]== '1']
    disco = df[df["GenreID"]== '2']
    blues = df[df["GenreID"]== '3']
    reggae = df[df["GenreID"]== '4']
    classical = df[df["GenreID"]== '5']
    rock = df[df["GenreID"]== '6']
    hiphop = df[df["GenreID"]== '7']
    country = df[df["GenreID"]== '8']
    jazz = df[df["GenreID"]== '9']
    return [pop, metal, disco, blues, reggae, classical, rock, hiphop, country, jazz]



def genreID_toString(genreID):
    dict = {0: "Pop", 1: "Metal", 2: "Disco", 3: "Blues", 4: "Reggae", 5: "Classical", 6: "Rock", 7: "Hiphop", 8: "Country", 9: "Jazz", } 
    return dict[genreID]


def knn_no_clustering(x,train_features,k=5):
    train_features, test_features = import_data("./data/GenreClassData_30s.txt")
    train_features = train_features.to_numpy()
    test_features = test_features.to_numpy()    

    
    dist = np.zeros(len(train_features)) 
    for i in range(len(train_features)):
        ref = {"mu": train_features[i][2:6]} 
        dist[i] = euclidean_distance(x,ref)

    lowest_k = np.argpartition(dist,k)[:k]
    lowest_k = train_features[lowest_k]
    genreIDs = lowest_k[:,1]
    #print()
    #print(genreIDs)
    counts = np.bincount(genreIDs.astype(np.int64))
    classification = np.argmax(counts)
    return classification

# Implements equation 3.16
def calc_mean(x,N):
    return (1/N)*sum(x)

# Implements equation 3.17
def calc_cov(x,mu,N):
    return (1/N)*sum(np.matmul((x - mu), np.transpose(x - mu)))




def cluster(xs):
    beta = 0
    M = 1
    mu1 = calc_mean(xs,len(xs))
    cov1 = calc_cov(xs,mu1,len(xs))
    lambda1 = {"mu": mu1,"sigma": cov1}
    for i in range(len(xs)): 
        dist += mahalanobis_distance(xs[i],lambda1)
    D1 = dist/len(xs)


    w = np.random.uniform(-delta,delta,size=len(mu1))
    lambdaNew = lambda1
    lambdaNew["mu"] += w
    lambdas = [lambda1,lambdaNew]

    while beta < 0.99:
        M = M+1 
    
    return lambda_list


def test():
    train_features, test_features = import_data("./data/GenreClassData_30s.txt")   
    data = train_features.to_numpy()
    data = divide_by_class(train_features)

    for i in range(len(data)):
        cluster(data[i])
test()

def test_no_clustering(x):
    #print(f"Genre of test: {genreID_toString(x[1])}")
   
    result = knn_no_clustering(x[2:6],train_features)
    
    #print(f"Resulting genre: {genreID_toString(result)}")
    return x[1] == result



train_features, test_features = import_data("./data/GenreClassData_30s.txt")
train_features = train_features.to_numpy()
test_features = test_features.to_numpy()    

counts = [0,0]
for i in range(len(test_features)):
    res = int(test_no_clustering(test_features[i]))
    counts[res] += 1

plt.bar(["Wrong","Correct"], counts)
plt.show()

