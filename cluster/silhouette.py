import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric


    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        scores = []
        num_obsv = X.shape[0]

        for i in range(num_obsv):

            a = 0
            b = np.inf

            obsv = np.reshape(X[i, :], (1, X.shape[1]))
            label = y[i]

            # CALCULATE a------------------------------------------------------------------------
            # Identify other points in observation i's cluster
            same_indices = np.where(y == label)[0]

            # Retrieve other points in observation i's cluster
            same_points = X[same_indices, :]


            # Calculate distances from observation i to all other points in its own cluster
            # and find the average.
            dists = cdist(obsv, same_points, self.metric).flatten().tolist()

            # If there are no other points in observation i's cluster, a = 0
            if len(dists) == 1:
                a = 0
            else:
                a = sum(dists)/(len(dists)-1) # Minus one because distance to itself = 0 and doesn't count


            # CALCULATE b--------------------------------------------------------------------------
            all_labels = set(y.tolist())

            for l in all_labels:
                if l != label:

                    # Identify points in a neighboring cluster
                    other_indices = np.where(y == l)[0]
                    other_points = X[other_indices, :]

                    # Calculate distances from observation i to other points in neighboring
                    # cluster and take the average 
                    dists = cdist(obsv, other_points, self.metric).flatten().tolist()
                    avg_dist = sum(dists)/(len(dists))

                    # Set b to be the avg distance from the closest neighboring cluster
                    if avg_dist < b:
                        b = avg_dist


                        
            # Calculate silhouette score
            score = (b-a)/max(b, a)

            scores.append(score)


        return np.array(scores)
