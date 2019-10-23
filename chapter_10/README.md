# Chapter 10: Grouping unlabeled items using k-means clustering


## k-means clustering
```text
Pros: Easy to implement
Cons: Can converge at local minima; slow on very large datasets 
Works with: Numeric values
```

k-means is an algorithm that will find k clusters for a given dataset. The number of
clusters k is user defined. Each cluster is described by a single point known as the
*centroid*. Centroid means it's at the center of all the points in the cluster.

The k-means algorithm works like this. First, the k centroids are randomly assigned
to a point. Next, each point in the dataset is assigned to a cluster. The assignment is done by finding the closest centroid and assigning the point to that cluster. After this step, the centroids are all updated by taking the mean value of all the points in that cluster.
pseudo code for k-Means algo:
```text
Algo:
Create k points for starting centroids (often randomly)
While any point has changed cluster assignment
 for every point in our dataset:
   for every centroid
     calculate the distance between the centroid and point
   assign the point to the cluster with the lowest distance 
 for every cluster calculate the mean of the points in that cluster
   assign the centroid to the mean
```

## Improving cluster performance with postprocessing
k-means has converged, but the cluster assignment isn't that great. The reason that k-means converged but we had poor clustering was that k-means converges on a local minimum, not a global minimum. (A local minimum means that the result is good but not necessarily the best possible. A global minimum is the best possible.)

We can use SSE(sum of squared error) to measure the quality of your cluster assignments. A lower SSE means that points are closer to their centroids, and you’ve done a better job of clustering.

## Bisecting k-means
pseudo code for bisecting k-Means algo:
```
Algo:
Start with all the points in one cluster
While the number of clusters is less than k
  for every cluster
    measure total error
    perform k-means clustering with k=2 on the given cluster
    measure total error after k-means has split the cluster in two
  choose the cluster split that gives the lowest error and commit this split
```
