import knn as knn
import matplotlib.pyplot as plt

def fetch_features():
    train_features, test_features = knn.import_data("./data/GenreClassData_30s.txt")
    train_features['GenreID'] = train_features['GenreID'].astype(str)
    classes = knn.divide_by_class(train_features)
    
    pop = classes[0]
    metal = classes[1]
    disco = classes[2]
    classical = classes[5]

    pop_features = pop.to_numpy()[:, 2:6]
    metal_features = metal.to_numpy()[:, 2:6]
    disco_features = disco.to_numpy()[:, 2:6]
    classical_features = classical.to_numpy()[:, 2:6]

    return pop_features, metal_features, disco_features, classical_features

def plot_histogram(genre, ax, color, label):
    N = 50
    
    ax[0][0].hist(genre[:, 0], bins = N, color = color, label = label)
    ax[0][0].set_title('Spectral rolloff mean')
    ax[0][0].legend()

    ax[0][1].hist(genre[:, 1], bins = N, color = color, label = label)
    ax[0][1].set_title('Mel-frequency cepstrum mean')
    ax[0][1].legend()

    ax[1][0].hist(genre[:, 2], bins = N, color = color, label = label)
    ax[1][0].set_title('Spectral centroid mean')
    ax[1][0].legend()

    ax[1][1].hist(genre[:, 3], bins = N, color = color, label = label)
    ax[1][1].set_title('Tempo')
    ax[1][1].legend()


def visualize_distribution():
    fig, ax = plt.subplots(2, 2)
    pop, metal, disco, classical = fetch_features()

    plot_histogram(pop, ax, 'blue', 'Pop')
    plot_histogram(metal, ax, 'orange', 'Metal')
    plot_histogram(disco, ax, 'green', 'Disco')
    plot_histogram(classical, ax, 'purple', 'Classical')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_distribution()