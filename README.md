# Build Your Own Regressor

Your task today is to create a new class, `MeanRegressor`, which implements a similar interface to a `sklearn` regressor like `sklearn.linear_model.LinearRegression`.  However, unlike a more sophisticated model implementation, your model does not actually take any of the feature variables into account, it just always predicts the mean of the target variable of the training data.

## Requirements

1. Build a class `MeanRegressor` that can be initialized
2. Write a method `MeanRegressor#fit(X, y)`:
    - `X` is a two-dimensional matrix (nested NumPy array, nested Python list, or Pandas dataframe) of data rows and features.  Your model will be ignoring it.
    - `y` is a list (NumPy array or Python list) representing the target variable
    - The model should determine the mean of `y` and store it, to be used in the `predict` method
    - This method does not return anything
3. Write a method `MeanRegressor#predict(X)`:
    - `X` is a two-dimensional matrix.  Your model will be ignoring its features, and only using the count of rows.
    - This method returns the mean of the training data for each row of `X`, i.e. a list containing the same number repeated as many times as necessary.
4. Write a method `MeanRegressor#score(X, y)`:
    - `X` is a two-dimensional matrix and `y` is a list of target variables
    - This method will compute the R<sup>2</sup> for how well the features of `X` are able to predict the target of `y`.  As a reminder, R<sup>2</sup> is calculated as 1 - `residual sum of squares`/`total sum of squares`, where `residual sum of squares` is the sum of all ((y_true - y_pred)<sup>2</sup>) and `total sum of squares` is the sum of all ((y_true - y_pred.mean())<sup>2</sup>).  So, if you are scoring with the same `y` that was used for the `fit`, you should expect the score to be exactly zero.

These requirements have test coverage.  To run the tests, run `pytest` in bash from the root of this repository.  There should be an initial print-out saying that you failed all 5 tests.  Re-run `pytest` as you implement the requirements, and eventually they should all pass.

## Hints

You will need at least one attribute (AKA member variable) to make your `MeanRegressor` work.  One option would be storing some information about model fit that you manually create, and another option would be instantiating a [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) and passing user inputs into it.

## Stretch Goals

If you have the previous requirements working and there is time remaining, consider these additional goals:

1. Check whether `X` and `y` are valid inputs and [raise appropriate exceptions](https://docs.python.org/3/tutorial/errors.html#raising-exceptionshttps://docs.python.org/3/tutorial/errors.html#raising-exceptions) if they are not.  For example, if the user tries to run `fit` with `X` having 5 rows and `y` having 10 target variables, produce a readable/understandable error that explains why this is invalid input.
2. See if you can reuse code from your Mod 2 project, and feed the King County housing data into your `MeanRegressor`.  How does this "dummy" model compare to your previous final model, in terms of R^2 and residuals?  Explore this with data visualizations.
