from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans


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
        self._models = [RandomForestClassifier(n_estimators=150)]#, XGBClassifier(booster='gbtree', learning_rate=0.1, max_depth=5, n_estimators=220), KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=100)]
        # self._models = [XGBClassifier(booster='gbtree', learning_rate=0.1, max_depth=5, n_estimators=220)]
        # self._models = [ KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=100)]
        # xgb = XGBClassifier(booster='gbtree', learning_rate=0.1, max_depth=5,
        #                     n_estimators=180)
        # xgb.fit(x_train, y_train)
        #
        # y_pred_xgb = xgb.predict(x_test)
        #
        # acc_xgb = accuracy_score(y_test, y_pred_xgb)
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

        for model in self._models:
            model.fit(X.astype('int'), y.astype('int'))

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
        # return self._models.predict(X)
        predicts = np.zeros((np.shape(X)[0], np.shape(self._models)[0]))
        for index, model in enumerate(self._models):
            predicts[:, index] = model.predict(X)
        predicts = np.sum(predicts, axis=1)
        number_of_votes = np.shape(self._models)[0]/2
        predicts[predicts <= number_of_votes] = 0
        predicts[predicts > number_of_votes] = 1
        return predicts

    def present(self, X, y):
        pretictions = self.predict(X)
        print(confusion_matrix(y, pretictions))
        # return self._models.score(X, y)
        acc = np.zeros((np.shape(self._models)[0]))
        print("precision_score :" +str(precision_score(y, pretictions)))
        print("f1_score :"+str(f1_score(y, pretictions)))
        for index, model in enumerate(self._models):
            acc[index] = model.score(X, y)
        return np.mean(acc)

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
