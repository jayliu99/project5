# Write your k-means unit tests here
import pytest
import numpy as np
from cluster import (
		KMeans, 
		Silhouette, 
		make_clusters,
		plot_clusters,
		plot_multipanel)


def test_kmeans():

	""" Helper function to check the correctness of kmeans clustering algorithm
		Tests:
			- kmeans with 4 clusters and scale=1 returns a plot similar to that of the provided example
			- kmeans returns an error if the provided k<2
			- kmeans returns an error if the number of observations < provided k
			- model can handle very high k (up to number of observations)
	"""

	clusters, labels = make_clusters(k=4, scale=1)

	# Test that kmeans returns a plot similar to that of the provided example (visual test)
	km = KMeans(k=4)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	plot_multipanel(clusters, labels, pred, scores)
	
	# Test that kmeans returns an error if k<2
	with pytest.raises(Exception):
		km = KMeans(k=0)
	with pytest.raises(Exception):
		km = KMeans(k=1)

	# Test that kmeans erturns an error if the number of observations < provided k
	km = KMeans(k=501)
	with pytest.raises(Exception):
		km.fit(clusters)

	# Test that model can handle very high k (visual test)
	km = KMeans(k=500)
	km.fit(clusters)
	pred = km.predict(clusters)
	scores = Silhouette().score(clusters, pred)
	plot_multipanel(clusters, labels, pred, scores)