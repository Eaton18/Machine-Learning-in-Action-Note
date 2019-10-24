# Chapter 2: Classifying with k-Nearest Neighbors

## 2.1 Classifying with distance measurements
```text
k-Nearest Neighbors
Pros: High accuracy, insensitive to outliers, no assumptions about data
Cons: Computationally expensive, requires a lot of memory
Works with: Numeric values, nominal values
```

The first machine-learning algorithm we’ll look at is k-Nearest Neighbors (kNN). It works like this: we have an existing set of example data, our training set. We have labels for all of this data)we know what class each piece of the data should fall into). When we’re given a new piece of data without a label, we compare that new piece of data to the existing data, every piece of existing data. We then take the most similar pieces of data (the nearest neighbors) and look at their labels. We look at the top k most similar pieces of data from our known dataset; this is where the k comes from. (k is an integer and it’s usually less than 20.) Lastly, we take a majority vote from the k most similar pieces of data, and the majority is the new class we assign to the data we were asked to classify.

**Pseudocode for k-NN algorithm**
```text
For every point in our dataset:
    calculate the distance between inX and the current point
    sort the distances in increasing order
    take k items with lowest distances to inX
    find the majority class among these items
    return the majority class as our prediction for the class of inX
```
