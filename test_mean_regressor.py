import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# these are just randomly generated numbers
df = pd.DataFrame(np.array([[0.03456305, 0.73344279, 0.83698467, 0.80022716],
                            [0.07772264, 0.59687277, 0.23005054, 0.94049597],
                            [0.65941026, 0.9521347, 0.98438789, 0.06182956],
                            [0.2296315, 0.7742189, 0.98567224, 0.05520582],
                            [0.1603394, 0.03856181, 0.83716034, 0.67256781],
                            [0.61952868, 0.16599499, 0.37051384, 0.71050288],
                            [0.28943516, 0.74314934, 0.87326874, 0.40335732],
                            [0.01063538, 0.09963898, 0.73835912, 0.82947293],
                            [0.88033203, 0.54813793, 0.79445131, 0.0892636],
                            [0.65128662, 0.79598075, 0.24360926, 0.78860746],
                            [0.86367495, 0.81685485, 0.80959694, 0.41850091],
                            [0.50656286, 0.28067404, 0.12122773, 0.31382457],
                            [0.86192731, 0.33515889, 0.91454857, 0.45355265],
                            [0.1084613, 0.56132593, 0.07804713, 0.78209782],
                            [0.34145706, 0.95703161, 0.89986143, 0.49346681],
                            [0.69771423, 0.26430632, 0.0236533, 0.78557031],
                            [0.38369573, 0.06686051, 0.74635996, 0.42010395],
                            [0.34493483, 0.48406464, 0.13753143, 0.01064768],
                            [0.67557827, 0.76489959, 0.77993042, 0.29599617],
                            [0.02988845, 0.98354185, 0.67689794, 0.70413554]]),
                  columns=["target", "a", "b", "c"])
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2019)


def test_constructor():
    """ Test to make sure that the MeanRegressor class exists, and that its
    constructor makes an instance of type MeanRegressor """
    from mean_regressor import MeanRegressor
    model = MeanRegressor()
    assert isinstance(model, MeanRegressor)

def test_fit():
    """ Test that the `fit` method does not return anything.  Because the
    internal behavior of the model is not specified, this test cannot actually
    check whether `fit` "worked" """
    from mean_regressor import MeanRegressor
    model = MeanRegressor()
    assert model.fit(X_train, y_train) == None

def test_predict():
    """ Test that the `predict` method returns the mean from the training data """
    from mean_regressor import MeanRegressor
    model = MeanRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    expected_predictions = np.repeat(0.437354358, len(X_test))
    assert np.allclose(predictions, expected_predictions)

def test_score_same_y():
    """ Test that the `score` method returns 0 if you use the training data,
    since there is no difference between our model and guessing the mean """
    from mean_regressor import MeanRegressor
    model = MeanRegressor()
    model.fit(X_train, y_train)
    r_squared = model.score(X_train, y_train)
    assert np.isclose(r_squared, 0.0)

def test_score_different_y():
    """ Test that the `score` method computes the correct score on the test data"""
    from mean_regressor import MeanRegressor
    model = MeanRegressor()
    model.fit(X_train, y_train)
    r_squared = model.score(X_test, y_test)
    assert np.isclose(r_squared, -0.10774219306207278)
