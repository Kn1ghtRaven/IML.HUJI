from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        # self._models = LogisticRegression(max_iter=1000)
        # self._models = SVC(kernel='rbf', probability=False)
        # self._model_names = "BRF SVM"
        self._models = RandomForestClassifier()
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # self._models.fit(X, y)
        self._models.fit(X.astype('int'), y.astype('int'))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._models.predict(X)

    def present(self, X, y):
        print(confusion_matrix(y, self.predict(X)))
        return self._models.score(X, y)

    def report(self, X_test, y_true):
        y_pred = self._models.predict(X_test)

        acc_rd_clf = accuracy_score(y_true, y_pred)
        conf = confusion_matrix(y_true, y_pred)
        clf_report = classification_report(y_true, y_pred)

        print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
        print(f"Confusion Matrix : \n{conf}")
        print(f"Classification Report : \n{clf_report}")

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pass
