# Chapter 13: Using principal component analysis to simplify data


## Dimensionality reduction techniques
A short list of other reasons we want to simplify our data includes the following:
- Making the dataset easier to use
- Reducing computational cost of many algorithms
- Removing noise
- Making the results easier to understand
There are dimensionality reduction techniques that work on labeled and unlabeled data.


## PCA(Principal Component Analysis)
The first new axis is chosen in the direction of the most variance in the data. The second axis is orthogonal to the first axis and in the direction of an orthogonal axis with the largest variance(The largest variation is the data telling us what’s most important). This procedure is repeated for as many features as we had in the original data.
```
Pros: Reduces complexity of data, indentifies most important features
Cons: May not be needed, could throw away useful information
Works with: Numerical values
```
Once we have the eigenvectors of the covariance matrix, we can take the top N eigenvectors. The top N eigenvectors will give us the true structure of the N most important features. We can then multiply the data by the top N eigenvectors to transform our data into the new space.

Pseudocode for transforming out data into the top N principal components would look like this:
```
Remove the mean
Compute the covariance matrix
Find the eigenvalues and eigenvectors of the covariance matrix
Sort the eigenvalues from largest to smallest
Take the top N eigenvectors
Transform the data into the new space created by the top N eigenvectors
```
