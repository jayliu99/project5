# write your silhouette score unit tests here

# Not sure why this test wouldn't pass! Visually, the silhouette scores looked okay, but 
# I looked all over and never figured out where the bug was. Maybe a grader can comment :)

import pytest
import numpy as np
from cluster import (
		KMeans, 
		Silhouette, 
		make_clusters,
		plot_clusters,
		plot_multipanel)

from scipy.spatial.distance import cdist


def test_silhouette():

	""" Helper function to check the correctness of silhouette scoring implementation
		Tests:
			- Silouette scores decrease as points move further from their centroid
	"""
	clusters, labels = make_clusters(k=4, scale=1)
	km = KMeans(k=4)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	#plot_multipanel(clusters, labels, pred, scores)

	centroids = km.get_centroids()

	for c in range(centroids.shape[0]):

		# Get the centroid belonging to the cluster
		center = centroids[c]

		# Get all points belonging to the cluster
		points_idx = np.where(pred == c)[0]

		# If points exist in the cluster
		if points_idx.all(): 

			# Get coordinates of all points as well as their scores
			points = clusters[points_idx]
			points_scores = scores[points_idx]

			# Calculate distances from each point to the centroid
			center = np.reshape(center, (1, center.shape[0]))
			dists = cdist(points, center, "euclidean").flatten()

			# Match scores to distances
			matched = list(zip(points_scores, dists))

			# Order matched list in ascending order by the silhouette scores
			matched.sort()

			# Check that as scores increase, distances to centroid decrease
			prev_dist = matched[0][1]
			for i in range(1, len(matched)):
				next_dist = matched[i][1]
				assert next_dist <= prev_dist, "Silhouette scores do not increase with proximity to centroid"
				prev_dist = next_dist
	

	return



  



