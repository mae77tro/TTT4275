import knn as knn
import numpy as np
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

# Plotting all genres over each other
def plot_histogram_overlay(genre, ax, color, label):
    N = 20
    
    ax[0][0].hist(genre[:, 0], bins = N, color = color, label = label, alpha = 0.8)
    ax[0][0].set_title('Spectral rolloff mean')
    ax[0][0].legend()

    ax[0][1].hist(genre[:, 1], bins = N, color = color, label = label, alpha = 0.8)
    ax[0][1].set_title('Mel-frequency cepstrum mean')
    ax[0][1].legend()

    ax[1][0].hist(genre[:, 2], bins = N, color = color, label = label, alpha = 0.8)
    ax[1][0].set_title('Spectral centroid mean')
    ax[1][0].legend()

    ax[1][1].hist(genre[:, 3], bins = N, color = color, label = label, alpha = 0.8)
    ax[1][1].set_title('Tempo')
    ax[1][1].legend()

# Plotting all features, per genre
def plot_feature_histogram(pop, metal, disco, classical, ax, fig, color, title, feature):
    N = 50
    
    # feature = 0: spectral rolloff mean
    # feature = 1: mfcc_mean
    # feature = 2: spectral centroid mean
    # feature = 3: tempo

    xmin = min(
        np.min(pop[:, feature]),
        np.min(metal[:, feature]),
        np.min(disco[:, feature]),
        np.min(classical[:, feature])
    )
    
    xmax = max(
        np.max(pop[:, feature]),
        np.max(metal[:, feature]),
        np.max(disco[:, feature]),
        np.max(classical[:, feature])
    )

    ax[0].hist(pop[:, feature], bins=N, color=color, alpha=0.8)
    ax[0].set(xlim=(xmin, xmax))
    ax[0].set_title('Pop')

    ax[1].hist(metal[:, feature], bins = N, color = color, alpha = 0.8)
    ax[1].set(xlim=(xmin, xmax))
    ax[1].set_title('Metal')

    ax[2].hist(disco[:, feature], bins = N, color = color, alpha = 0.8)
    ax[2].set(xlim=(xmin, xmax))
    ax[2].set_title('Disco')

    ax[3].hist(classical[:, feature], bins = N, color = color, alpha = 0.8)
    ax[3].set(xlim=(xmin, xmax))
    ax[3].set_title('Classical')

    fig.suptitle(title)


def visualize_distribution():
    pop, metal, disco, classical = fetch_features()
    
    fig0, ax0 = plt.subplots(4, 1)
    plot_feature_histogram(pop, metal, disco, classical, ax0, fig0, 'blue', 'Spectral Rolloff Mean Histogram', 0)
    
    fig1, ax1 = plt.subplots(4, 1)
    plot_feature_histogram(pop, metal, disco, classical, ax1, fig1, 'orange', 'Mel-frequency Cepstrum Mean Histogram', 1)

    fig2, ax2 = plt.subplots(4, 1)
    plot_feature_histogram(pop, metal, disco, classical, ax2, fig2, 'purple', 'Spectral Controid Mean Histogram', 2)

    fig3, ax3 = plt.subplots(4, 1)
    plot_feature_histogram(pop, metal, disco, classical, ax3, fig3, 'green', 'Tempo Histogram', 3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_distribution()