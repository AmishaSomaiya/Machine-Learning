# HW0

## Setup

**Before you start working on any problem** make sure you follow instructions of setting up environment. You can find them in [root folder README](../../README.md). If you have any problems or questions about installing conda, whether you should use anaconda vs miniconda, why are we even using conda etc. feel free to reach out to TAs, stack overflow or Google.

Remaining instructions for this problem are written assuming that you have successfully setup environment `cse446`, installed all packages in it (including `pip install -e .`) and have activated it (`conda activate cse446`).

## Vanilla vs numpy

In this problem you will explore how numpy can be used to get significant speedups over basic python.
You will implement a solution to another problem on hw0.
To start with this problem look at [`vanilla_vs_numpy.py`](./vanilla_vs_numpy.py), and implement `numpy_solution` and `vanilla_solution` functions.

After this you will need to test whether your implementations are correct with `inv test`.
If you pass all tests (you should see no error messages, `E`s or `F`s, only `.`s), run the file you just worked in. For example if you are in root directory of provided `.zip` file then you can run `python homeworks/vanilla_vs_numpy/vanilla_vs_numpy.py`. This will generate a plot you will need to answer a question in your written submission.

Purpose of this problem is to get you familiar with numpy.
Even as you finish the problem we advise you to bookmark its documentation, as you will be using a lot in this course.
