import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_features(features: np.array, class_embeddings: np.array, labels: np.array)->None:
    '''
        Plot the features and the class embeddings

        Args:
         - features: all features (n_samples, feature_dims)
         - class_embeddings: all class embedings (num_class,)
         - labels: all labels ranging from 0 to num_class-1 (n_samples,)
    '''
    num_class = class_embeddings.shape[0]

    # Learn features
    class_labels = -1*np.ones(num_class) # dummy label idx
    X = np.concatenate((features.numpy(),class_embeddings),axis=0)
    Y = np.concatenate((labels.numpy(),class_labels),axis=0)

    np.save('./train_features.npy',X.numpy())
    np.save('./train_labels_idx.npy',Y.numpy())

    X = X/np.linalg.norm(X,2,axis=-1,keepdims=True)

    tsne = TSNE(n_components = 2)
    tsne.fit_transform(X)
    lowdim_X = tsne.embedding_
    c_map_1 = plt.get_cmap('tab10')
    for class_idx in range(10):
        plt.scatter(lowdim_X[Y==class_idx,0],lowdim_X[Y==class_idx,1],
                    marker='x',
                    s=3,
                    color=c_map_1(class_idx))
    for class_idx in range(10,150):
        plt.scatter(lowdim_X[Y==class_idx,0],lowdim_X[Y==class_idx,1],
                    marker='.',
                    s=3,
                    color='grey',
                    alpha=0.1)
    for class_idx in range(10):
        plt.scatter(lowdim_X[-num_class+class_idx,0],lowdim_X[-num_class+class_idx,1],
                    marker='*',
                    s=10,
                    color=c_map_1(class_idx))
    for class_idx in range(10,150):
        plt.scatter(lowdim_X[-num_class+class_idx,0],lowdim_X[-num_class+class_idx,1],
                    marker='*',
                    s=10,
                    color='grey')
    plt.savefig('./figures/tsne-taskall-featureall.png',bbox_inches='tight',dpi=1200)
    plt.show()
    plt.clf()

    c_map_2 = plt.get_cmap('cool')
    for class_idx in range(10):
        plt.scatter(lowdim_X[Y==class_idx,0],lowdim_X[Y==class_idx,1],
                    marker='x',
                    s=3,
                    color=c_map_1(class_idx))
    for class_idx in range(10,150):
        plt.scatter(lowdim_X[Y==class_idx,0],lowdim_X[Y==class_idx,1],
                    marker='.',
                    s=3,
                    color=c_map_2(class_idx/150),
                    alpha=0.1)
    for class_idx in range(10):
        plt.scatter(lowdim_X[-num_class+class_idx,0],lowdim_X[-num_class+class_idx,1],
                    marker='*',
                    s=10,
                    color=c_map_1(class_idx))
    for class_idx in range(10,150):
        plt.scatter(lowdim_X[-num_class+class_idx,0],lowdim_X[-num_class+class_idx,1],
                    marker='*',
                    s=10,
                    color=c_map_2(class_idx/150),
                    alpha=0.1)
    plt.savefig('./figures/tsne-taskall-feature1.png',bbox_inches='tight',dpi=1200)
    plt.show()
    plt.clf()