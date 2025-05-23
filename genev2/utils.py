from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(X, perplexity=30, random_state=0):
    """
    Compute a 2D t-SNE embedding of X and display a scatter plot.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data to embed.
    perplexity : float
        The perplexity parameter for t-SNE.
    random_state : int
        Seed for t-SNE randomness.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title("t-SNE Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()