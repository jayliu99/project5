import numpy as np
from scipy.spatial.distance import cdist
import random

random.seed(25) 

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k 
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = [] # Stores centroids
        self.labels = [] # Stores cluster labels for all points

        assert k>1, "Number of clusters to form (k) must be at least 2"


    def _reinit_labels(self):
        """
        resets cluster labels for all points

        """

        self.labels = []

        return



    def _assign(self, num_obsv: int, dists: np.ndarray, mat: np.ndarray):
        """
        assigns observations in dataset to their closest cluster by referencing a
        precalculated matrix of distances between cluster centroids and observations

        inputs: 
            num_obsv: int
                The number of observations in dataset 
                
            dists: np.ndarray
                A 2D matrix where the rows are observations and columns are clusters.
                The i,jth entry represents the Euclidean distance between observation i 
                and the centroid of cluster j

            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features.
        """
        for m in range(num_obsv):

            # Retrieve distances between observation m and current centroids
            m_dist = dists[m].tolist()

            # Figure out which centroid is currently closest to observation m
            c = m_dist.index(min(m_dist))

            # Label observation m as belonging to the cluster of the centroid to which it's closest
            self.labels.append(c)

        return



    def _recalc_centroids(self, num_fts: int, mat: np.ndarray):
        """
        helper function for the kmeans clustering algorithm that computes new centroids
        based on current cluster assignments

        inputs: 
                
            num_fts: int
                The number of features in each observation

            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

        outputs:
            np.ndarray
                a 2D array with updated centroids for the model
        """
        new_centroids = np.empty(shape=(self.k, num_fts))

        for c in range(self.k):

            points_indices = np.where(np.asarray(self.labels) == c)[0]

            # If no points have been assigned to this cluster, the centroid doesn't
            # have to be updated.
            if len(points_indices) == 0:
                new_centroids[c] = self.centroids[c]

            else:

                # Calculate the new centroid in each cluster
                points = mat[points_indices]
                cluster_size = points.shape[0]
                new_centroids[c] = np.sum(points, axis=0)/cluster_size


        # Return the new centroids
        return new_centroids



    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        self._reinit_labels() # Reinitialize empty labels in case function is run multiple times

        num_obsv = mat.shape[0]
        num_fts = mat.shape[1]

        assert self.k <= num_obsv, "Number of clusters to form exceeds number of observations in dataset"

        # Initialize an empty matrix (k X m) to store centroid locations
        self.centroids = np.empty(shape=(self.k, num_fts))

        # Randomly initialize k centroids
        for i in range(self.k):
            self.centroids[i] = mat[random.randint(0, num_obsv-1)]


        iter_counter = 0

        while True:

            # Compute initial error
            old_error = self.get_error(mat)

            # Reset labels
            self._reinit_labels() 

            # Calculate Euclidean distance between all points and existing clusters.
            dists = cdist(mat, self.centroids, self.metric)

            # Iterate through observations, assigning each to the closest cluster
            self._assign(num_obsv, dists, mat)

            # Recalculate centroids after reassigning all points
            self.centroids = self._recalc_centroids(num_fts, mat)

            iter_counter+=1

            # Once we have iterated a maximum number of times OR stopped improving, end loop.
            if (iter_counter >= self.max_iter) or ((old_error - self.get_error(mat)) < self.tol):
                break

        return



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        assert self.centroids.size != 0, "Model must be fit before running predict."

        predictions = []

        # Compute Euclidean distance from each point to the model's existing centroids
        dists = cdist(mat, self.centroids, self.metric)

        num_points = mat.shape[0]

        # Iterate through points, assigning each to the closest cluster
        for m in range(num_points):
            m_dists = dists[m].tolist()
            c = m_dists.index(min(m_dists))
            predictions.append(c)

        return np.array(predictions)



    def get_error(self, mat) -> float:
        """
        returns the final mean-squared error of the fit model

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            float
                the mean-squared error of the fit model
        """
        errors = []

        for i in range(self.k):

            points_indices = np.where(np.asarray(self.labels) == i)[0]

            # If no points are assigned to the cluster, don't add anything to the error
            if len(points_indices) == 0:
                pass

            else:
                points = mat[points_indices]
                cluster_size = points.shape[0]
                centroid = np.reshape(np.sum(points, axis=0)/cluster_size, (1, mat.shape[1]))

                # Calculate the distance from each point to its corresponding centroid. 
                dists = cdist(points, centroid, self.metric)

                # Square and store the distances
                errors = errors + [x**2 for x in dists.flatten().tolist()]

        # If points have not been assigned to clusters, return a large error
        if len(errors) == 0:
            return np.inf

        else:
            # Average the squared distances.
            return sum(errors)/len(errors)




    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids






