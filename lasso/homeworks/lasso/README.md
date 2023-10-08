# Lasso

## Coordinate Descent Lasso
In this problem you will implement a Coordinate Descent Algorithm with Lasso regularization.
Start by looking into the file [coordinate_descent_algo.py](./coordinate_descent_algo.py), to see what functions you will need to implement. You should start with any of `step`, `loss`, `precalculate_a` or `convergence_criterion`. Once you have completed them, implement `train` and then `main` function.

For the `main` function there isn't a lot of direction, as this function should repeatedly call `train` and record various quantities around non zero entries in returned weight vector.
Lastly it should generate a plot for both parts a and b of the problem.
We will not test this function, so you can do it in any way you would like, but `plt.savefig(filename)` followed by `plt.cla()` and `plt.clf()` allows you to directly save a plot to a file.
Make sure to read hints in the pdf about implementation.Hot-starting weight vector might save you a lot of time.

Lastly when implementing `main` and checking for a resulting feel free to increase `convergence_delta` value in `train`. Anything around `0.1, 0.01` should run quite fast.
**For final submission we ask you to use `convergence_delta=1e-4` however.**

As in hw1 to run file for the problem do: `python homeworks/lasso/coordinate_descent_algo.py` from the root directory of provided zip file.

## Crime Data Lasso
In this problem you will use algorithm from Coordinate Descent Lasso problem and apply it on real-world social data.
Before you dive deep into writing code to generate various plots, you will need to look at the dataset and think through what can go wrong when using it in Machine Learning pipeline.
This is important, because if you will ever work in industry and make decisions about human lives you want to ensure your model is fair and robust.

Start by looking into the file [crime_data_lasso.py](./crime_data_lasso.py).
You will notice that there is only a single `main` function that you need to fill in.
You should not return anything, but create and save all of the plots in the problem.

The reason why we ask you to do so in a single function is because you should **loop through lambdas only once** and log information for **all of the sub-problems at once**.
This way our code will be faster.

To run the problem do: `python homeworks/lasso/crime_data_lasso.py` from the root directory of provided zip file, and then add the plots to your written submission.
