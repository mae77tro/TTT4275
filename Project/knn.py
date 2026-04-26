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

    train_features = train[["Track ID", "GenreID", "spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo", "mfcc_1_std", "spectral_contrast_mean", "spectral_contrast_var"]]
    test_features = test[["Track ID", "GenreID", "spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo", "mfcc_1_std", "spectral_contrast_mean", "spectral_contrast_var"]] 
    
    return train_features, test_features

def divide_by_class(df):
    pop = df[df["GenreID"]== 0]
    metal = df[df["GenreID"]== 1]
    disco = df[df["GenreID"]== 2]
    blues = df[df["GenreID"]== 3]
    reggae = df[df["GenreID"]== 4]
    classical = df[df["GenreID"]== 5]
    rock = df[df["GenreID"]== 6]
    hiphop = df[df["GenreID"]== 7]
    country = df[df["GenreID"]== 8]
    jazz = df[df["GenreID"]== 9]
    return [pop, metal, disco, blues, reggae, classical, rock, hiphop, country, jazz]



def genreID_toString(genreID):
    dict = {0: "Pop", 1: "Metal", 2: "Disco", 3: "Blues", 4: "Reggae", 5: "Classical", 6: "Rock", 7: "Hiphop", 8: "Country", 9: "Jazz", } 
    return dict[genreID]


def knn_no_clustering(x,clusters,k=5):
    clusters = np.array(clusters)
    
    dist = np.zeros(len(clusters)) 
    for i in range(len(clusters)):
        ref = {"mu": clusters[i][2:6]} 
        dist[i] = euclidean_distance(x,ref)

    lowest_k = np.argpartition(dist,k)[:k]
    lowest_k = clusters[lowest_k]
    genreIDs = lowest_k[:,1]
    counts = np.bincount(genreIDs.astype(np.int64))
    classification = np.argmax(counts)
    return classification

# Implements equation 3.16
def calc_mean(x,N):
    return (1/N)*sum(x)

# Implements equation 3.17
def calc_cov(x,mu,N):
    diff = x- mu
    new_cov = (diff.T @ diff)/N
    new_cov += np.eye(len(new_cov))*1e-6 # Making sure the covariance matrix will always be invertible
    return new_cov

# This function is based on the clustering procedure from the compendium on page 80-81
def cluster(data):
    genreID = data[0][1]
    # Starting with M = 1
    xs = data[:,2:6].astype(float)
    M = 1
    mu1 = calc_mean(xs,len(xs))
    cov1 = calc_cov(xs,mu1,len(xs))
    lambda1 = {"mu": mu1,"sigma": cov1}
    
    #Calculating distance
    dist = 0
    for i in range(len(xs)): 
        dist += mahalanobis_distance(xs[i],lambda1)
    D1 = dist/len(xs)
    lambdas = [lambda1]
    
    Dms = [D1]
    cluster_classifications = np.zeros(len(xs),dtype=int)
    #num_clusters = 3 #Forcing 5 clusters
    #for i in range(num_clusters):
    while True:
        # Make a new cluster by adding random value to mean
        cluster_classifications_old = cluster_classifications
        M = M+1

        counts = np.bincount(cluster_classifications_old.astype(int)) 
        biggest_cluster = np.argmax(counts) 
        orig_lambda = lambdas[biggest_cluster] 
        delta = 0.05*np.sqrt(np.diag(orig_lambda["sigma"]))

        #Creating two clusters from the old biggest one, (Just moving one cluster by w like said in the algorithm from the book didn't work that well) 
        w = np.random.uniform(-delta,delta,size=len(mu1))
        l1 = {"mu": orig_lambda["mu"]-w, "sigma": orig_lambda["sigma"] } 
        l2 = {"mu": orig_lambda["mu"]+w, "sigma": orig_lambda["sigma"] } 
        lambdas[biggest_cluster] = l1
        lambdas.append(l2)
        
        # Looping over clusters to find optimal amount of clusters  
        Dm_q = 10e10
        cluster_classifications = np.zeros(len(xs))
        q = 1
        while True:
            Dm_last_q = Dm_q
            Dm_q_sum = 0
            
            #Dividing into cllusters
            for i in range(len(xs)):
                shortest_dist = np.inf

                dists = [mahalanobis_distance(xs[i],l) for l in lambdas]
                closest = np.argmin(dists)
                cluster_classifications[i] = closest

                Dm_q_sum += dists[closest]

            Dm_q = Dm_q_sum/len(xs)
            beta_1 = (Dm_last_q-Dm_q)/Dm_last_q
            if beta_1 < 0.001:
                break

            # Updating mus and sigmas for each cluster
            for i in range(len(lambdas)):
                sub_xs = xs[cluster_classifications==i]
                if len(sub_xs) != 0:
                    new_mu = calc_mean(sub_xs,len(sub_xs))
                    new_cov = calc_cov(sub_xs,new_mu,len(sub_xs))
                    lambdas[i] = {"mu": new_mu, "sigma": new_cov} 
                else:
                    #No one chose this cluster :(
                    pass
        
        Dms.append(Dm_q)
        beta = Dms[-1]/Dms[-2]
        if beta >= 0.999:
            break
    return genreID, lambdas


def knn_with_clustering(x,clusters,k=5):
    dist = np.zeros(len(clusters))
    for i in range(len(clusters)):
        dist[i] = mahalanobis_distance(x,clusters[i])
    
    clusters_array = np.array(clusters)
    lowest_k_indices = np.argpartition(dist,k)[:k]
    
    genreIDs = [clusters[i]["genre"] for i in lowest_k_indices]
    #print()
    #print(genreIDs)
    counts = np.bincount(genreIDs)
    classification = np.argmax(counts)
    return classification

def test_no_clustering(x,train_features):
    #print(f"Genre of test: {genreID_toString(x[1])}")
   
    result = knn_no_clustering(x[2:6],train_features)
    
    #print(f"Resulting genre: {genreID_toString(result)}")
    return x[1] == result
        

# To Daniel: Fixed so I could import knn.py-functions without running the testing code.
if __name__ == "__main__":
    clusters, test_features = import_data("./data/GenreClassData_30s.txt")
    data_no_clustering = clusters.to_numpy()
    test_features = test_features.to_numpy()    
    
    # Normalizing the data:
    train_xs = data_no_clustering[:,2:6].astype(float)
    test_xs = test_features[:,2:6].astype(float)

    train_mean = np.mean(train_xs,axis=0)
    train_std = np.std(train_xs,axis=0)
    train_std[train_std == 0] = 1e-6 # Adding epsilon to not get singular matrix
    
    train_xs_scaled = (train_xs-train_mean)/train_std
    test_xs_scaled = (test_xs-train_mean)/train_std

    data_no_clustering[:,2:6] = train_xs_scaled
    test_features[:, 2:6] = test_xs_scaled

    clusters.iloc[:, 2:6] = train_xs_scaled

    # k-nn without clustering
    counts_without_clustering = [0,0]
    for i in range(len(test_features)):
        res = int(test_no_clustering(test_features[i],data_no_clustering))
        counts_without_clustering[res] += 1
    print(f"Correct without clustering: {counts_without_clustering[1]}")
    print(f"Wrong: {counts_without_clustering[0]}")   

    plt.title("Without clustering")
    plt.bar(["Wrong","Correct"], counts_without_clustering)


    # k-nn with clustering
    data = clusters
    num_genres = 10

    data = divide_by_class(clusters)
    #print("Data: ")
    #print(data) 
    clusters = [None]*num_genres
    for i in range(len(data)):
        genreID, new_cluster = cluster(data[i].to_numpy())
        #print("genreID") 
        genreID = int(genreID)
        clusters[genreID] = new_cluster
    #print("Clusters: ")
    #print(clusters)


    #Adding genreid as the last item to all the datapoints
    data_with_genreid = []
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            newdata = clusters[i][j]
            entry = {
                "mu": newdata["mu"],
                "sigma": newdata["sigma"],
                "genre": i
            }

            #print("Row: ") 
            #print(row)
            data_with_genreid.append(entry)
        print(f"Clusters for genre {i}: {j}")

    counts_with_clustering = [0,0]
    for i in range(len(test_features)):
        res = int(knn_with_clustering(test_features[i][2:6],data_with_genreid,k=1))
        correct = res == test_features[i][1]
        counts_with_clustering[int(correct)] += 1

    print(f"Correct with clustering: {counts_with_clustering[1]}")
    print(f"Wrong: {counts_with_clustering[0]}")

    plt.figure() 
    plt.title("With clustering")
    plt.bar(["Wrong","Correct"],counts_with_clustering)

    plt.show()