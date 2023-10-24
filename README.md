# Machine-Learning

This repo contains Machine Learning implementations using Python for the following :

- Neural Networks for MNIST classification (FROM SCRATCH, WITHOUT USING TORCH.NN) 
- Kernel Ridge Regression using the polynomial and RBF kernels
- Regularized Logistic Regression for Binary Classification for MNIST
- Regularized Polynomial Regression using Numpy
- Ridge Regression on MNIST (WITHOUT SCIKIT-LEARN)
- Co-ordinate Descent Algorithm for Lasso
- KMeans Clustering using Lloyd's Algorithm 
- Principal Component Analysis
- Comparison of execution times using native python and Numpy vector form
- Empirical study of Central Limit Theorem in code
- simCLR implementation with DCL Loss






## Regularized Polynomial Regression

Regularized polynomial regression is implemented in polyreg.py. All matrices are 2D NumPy arrays in the implementation.  
__init__(degree=1, regLambda=1E-8) : constructor with arguments of d and λ
fit(X,Y): method to train the polynomial regression model
predict(X): method to use the trained polynomial regression model for prediction
polyfeatures(X, degree): expands the given n × 1 matrix X into an n × d matrix of polynomial features of degree d. The returned matrix will not include the zero-th power. 

The polyfeatures(X, degree) function maps the original univariate data into its higher order powers. Specifically, X will be an n × 1 matrix. and this function will return the polynomial expansion of this data, a n × d matrix. This function will not add in the zero-th order feature
(i.e., x0 = 1). The x0 feature is added separately, outside of this function, before training the model.

By not including the x0 column in the matrix polyfeatures(), this allows the polyfeatures function
to be more general, so it could be applied to multi-variate data as well. (If it did add the x0 feature, we’d
end up with multiple columns of 1’s for multivariate data.)

Also, the resulting features will be badly scaled if used in raw form. For example, with
a polynomial of degree d = 8 and x = 20, the basis expansion yields an absolutely huge difference in range. Consequently, standardize the data in fit() after performing the polynomial feature expansion and before solving linear regression. Apply the same standardization transformation in predict() before applying it to new
data. Run plot_polyreg_univariate.py to test implementation, this will plot the learned
function. In this case, the script fits a polynomial of degree d = 8 with no regularization λ = 0. From
the plot, see that the function fits the data well, but will not generalize well to new data points.
Increase the amount of regularization to see the effect on the function.


## Ridge Regression on MNIST (WITHOUT SCIKIT-LEARN)

ridge_regression.py implements a regularized least squares classifier for the MNIST dataset to classify handwritten images of numbers between 0 to 9. the regularized least squares objective is minimized using a linear classifier : 1-layer neural network. The model is trained to estimate on the MNIST training data with λ = 10−4 and make label predictions on test data. 


## LASSO
Co-ordinate Descent Algorithm for Lasso

The coordinate descent algorithm to solve the LASSO problem is implemented in coordinate_descent_algo.py. 
Matrix libraries are used for matrix operations instead of for loops. Quantities like ak are precomputed to speed up the algorithm. The objective value is nonincreasing with each step. The stopping condition is when no element
of w changes by more than some small δ during an iteration.  The LASSO problem is solved on the same dataset for many values of λ. This is called a regularization path. This is done by starting at a large λ, and then for each consecutive solution, initializing the algorithm with the previous solution, decreasing λ by a constant ratio (e.g., by a factor of 2). A benefit of the Lasso is that if many features are irrelevant for predicting y, the Lasso can be used to enforce a sparse solution, effectively differentiating between the relevant and irrelevant features.

Trying the above coded LASSO implementation with synthetic data.Solved multiple Lasso problems on a regularization path, starting at
λmax where no features are selected and decreasing λ by a constant ratio (e.g., 2) until
nearly all the features are chosen.  For each value of λ tried, record values for false discovery rate (FDR) (number of incorrect
nonzeros in wb/total number of nonzeros in wb) and true positive rate (TPR) (number of correct nonzeros
in wb/k). Note: for each j, wbj is an incorrect nonzero if and only if wbj 6= 0 while wj = 0. In plot 2, plot
these values with the x-axis as FDR, and the y-axis as TPR.
Note that in an ideal situation we would have an (FDR,TPR) pair in the upper left corner.

Next, test LASSO implementation on crime : real dataset. 
This stores the data as Pandas DataFrame objects. DataFrames are similar to Numpy arrays but more flexible;
unlike arrays, DataFrames store row and column indices along with the values of the data. Each column of a
DataFrame can also store data of a different type (here, all data are floats). Few commands for working with Pandas :

df.head() # Print the first few lines of DataFrame df.

df.index # Get the row indices for df.

df.columns # Get the column indices.

df[``foo''] # Return the column named ``foo''.

df.drop(``foo'', axis = 1) # Return all columns except ``foo''.

df.values # Return the values as a Numpy array.

df[``foo''].values # Grab column foo and convert to Numpy array.

df.iloc[:3,:3] # Use numerical indices (like Numpy) to get 3 rows and cols.


The data consist of local crime statistics for 1,994 US communities. The response y is the rate of violent crimes
reported per capita in a community. The name of the response variable is ViolentCrimesPerPop, and it is held
in the first column of df_train and df_test. There are 95 features. These features include many variables.
Some features are the consequence of complex political processes, such as the size of the police force and other
systemic and historical factors. Others are demographic characteristics of the community, including self-reported
statistics about race, age, education, and employment drawn from Census reports.

Training a model on this dataset suggests a degree of correlation between a community’s demographics and the rate at which a community experiences and reports violent crime. The dataset is split into a training and test set with 1,595 and 399 entries, respectively. 


## Neural Networks for MNIST classification (FROM SCRATCH, WITHOUT USING TORCH.NN) 
A shallow but wide network, and a narrow but deeper network, both are implemented from scratch. For both architectures, use d to refer to the number of input features (in MNIST, d = 282 = 784), hi
to refer to the dimension of the i-th hidden layer and k for the number of target classes (in MNIST, k = 10).
For the non-linear activation, we use ReLU. 

Weight Initialization
Consider a weight matrix W ∈ R
n×m and b ∈ R
n. Note that here m refers to the input dimension and n to the
output dimension of the transformation x 7→ W x + b. Define α = 1/√m
. Initialize all weight matrices and
biases according to Unif(−α, α).


Adam optimizer is used for training. Adam is a more advanced form of gradient
descent that combines momentum and learning rate scaling. It often converges faster than regular gradient
descent in practice. We use cross entropy for the loss function and ReLU for the non-linearity. 



## Kernel Ridge Regression using the polynomial and RBF kernels
Using leave-one-out cross validation, we find a good λ and hyperparameter settings for the
kernels by search implementation WITHOUT USING SCIKIT-LEARN.




## Regularized Logistic Regression

Here we consider the MNIST dataset, for binary classification. Specifically, the task is to determine
whether a digit is a 2 or 7. Here, let Y = 1 for all the “7” digits in the dataset, and use Y = −1 for “2”.
We will use regularized logistic regression. The regularized negative log likelihood objective function is used. The offset term b is not regularized.  For all experiments, use λ = 10−1. Gradient Descent is implemented with an initial iterate of all zeros. Several values of step sizes are tried to find one that appears to make convergence on the training set as fast as possible.  We repeat above process for Stochastic Gradient Descent with batch size 1 and 100. 


## KMeans Clustering - Lloyd's Algorithm 

Implementation of Lloyd's algorithm without using scikit learn.



## Principal Component Analysis
PCA is implemented on MNIST dataset and the digits are reconstructed in the dimensionality-reduced PCA basis.
The PCA basis is computed using the training dataset only, and evaluated the quality of the basis on the test set,
similar to the k-means reconstructions above. We have ntrain = 50,000 training examples of size 28×28.. 



## Vanilla vs Numpy


1. Comparison of execution times using native python and Numpy vector form.
2. Two random variables X and Y have equal distributions if their CDFs, FX and FY , respectively, are equal,
i.e. for all x, |FX(x) − FY (x)| = 0. The central limit theorem says that the sum of k independent, zero-mean,
variance 1/k random variables converges to a (standard) Normal distribution as k tends to infinity. This phenomenon is studied empirically here. 


## simCLR implementation with DCL Loss

This implementation addresses the challenge of unavailability of annotations in real 
world vision datasets by generating representations by extracting useful features 
from raw unlabeled data. The project self-implements simCLR a popular 
framework for self-supervised contrastive learning for generating 
representations. This implementation is then evaluated on downstream tasks of  Linear Probing and Fine-Tuning. simCLR is computationally expensive and the 
 original paper implementation uses a default batch size of 4092 and 32 TPUs for 
good performance. Another issue with simCLR is the positive-negative 
coupling effect in the InfoNCE loss, due to which the model performance  degrades at sub-optimal hyperparameters like small batch sizes. This implementation 
addresses these issues by applying Decoupled Contrastive Learning (DCL) loss 
instead of simCLR loss. The final project implementation of simCLR with DCL 
loss with Resnet50 backbone at 100 epochs achieves a top1 accuracy of 84.84% 
at batch size of 32 for linear evaluation on CIFAR10 dataset. This outperforms 
the top1 accuracy of 81.66% at the same batch size of 32 with simCLR loss and 
has comparable performance to previously obtained top1 accuracy of 85.08% at 
higher batch size of 128. Thus, the final project implementation of simCLR with 
DCL efficiently generates vision representations at lower batch sizes with 
computational savings and good performance metrics that can be used for 
several downstream vision tasks.


	
