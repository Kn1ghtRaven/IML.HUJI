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
import torch as tr
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, X) -> AgodaCancellationEstimator:
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
        self.criterion = nn.BCELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()
        LEARNING_RATE = 0.0001

        # net = cancelPred(COLS)

        self.device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')
        self.net = cancelPred(len(X.columns), self.device)
        self.optimizer = tr.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.net.to(self.device)
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
        self.net.fit(X, y)
        #this is for the NN
        for epoch in range(self.net.epochs):
            running_loss = 0.0
            for i, data in enumerate(self.net.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.net.device), data[1].to(self.net.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                # print(outputs.min())
                # print(labels.reshape(outputs.shape))
                # print("pasten")
                loss = self.criterion(outputs, labels.reshape(outputs.shape))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                    running_loss = 0.0


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
        #this is for the NN
        self.net.preper_to_check_acc(X, y)
        correct_pred = 0
        total = 0
        preds = []
        y_labels = []

        with tr.no_grad():
            for data in self.net.testloader:
                images, labels = data
                outputs = self.net(images)
                predictions = tr.round(outputs)
                # print(predictions.reshape(labels.shape))
                # print(labels)
                # _, predictions = tr.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions.reshape(
                        labels.shape)):
                    # print(label, prediction)
                    preds.append(prediction.item())
                    y_labels.append(label.item())
                    if label == prediction:
                        correct_pred += 1
                    total += 1

        # print accuracy for each class
        accuracy = 100 * float(correct_pred) / total
        f1_score_net = float(correct_pred)/ float(correct_pred+ 0/5*(total - correct_pred))
        print('accuracy of the net is :%f' % accuracy)
        # print('f1_score of the net is :%f' % f1_score_net)
        print('f1 score by sklrean : %f' % f1_score(y_labels, preds))
        print(confusion_matrix(y_labels, preds))

        return np.mean(acc)

    # def report(self, X_test, y_true):
    #     y_pred = self._models.predict(X_test)
    #
    #     acc_rd_clf = accuracy_score(y_true, y_pred)
    #     conf = confusion_matrix(y_true, y_pred)
    #     clf_report = classification_report(y_true, y_pred)
    #
    #     print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
    #     print(f"Confusion Matrix : \n{conf}")
    #     print(f"Classification Report : \n{clf_report}")

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
class cancelPred(nn.Module):

    def __init__(self, features_no, device, inner_neutrons=400, LEARNING_RATE = 0.0001, EPOCHS = 14, BATCH_SIZE = 50):
        super().__init__()
        self.act = nn.Sigmoid
        self.inner_nurons = inner_neutrons
        self.learning_rate_ = LEARNING_RATE
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(features_no, self.inner_nurons),
            nn.Dropout(),
            self.act(),
            nn.Linear(self.inner_nurons, self.inner_nurons),
            nn.Dropout(),
            self.act(),
            nn.Linear(self.inner_nurons, self.inner_nurons),
            nn.Dropout(),
            self.act(),
            nn.Linear(self.inner_nurons, 1),
            self.act()
            )

    def forward(self, x):
        return self.fc(x)

    def fit(self, X, y):
        # train_x = tr.Tensor(x_train.values.astype(np.float32)).to(tr.float32).to(device)
        train_x = tr.tensor(X.values).to(tr.float32).to(self.device)
        train_y = tr.Tensor(y).float().to(self.device)
        trainset = TensorDataset(train_x, train_y)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def preper_to_check_acc(self, X, y):
        # test_x = tr.Tensor(x_test.values.astype(np.float32)).to(tr.float32)
        test_x = tr.tensor(X.values).to(tr.float32)
        test_y = tr.tensor(y.values).float()
        self.testset = TensorDataset(test_x, test_y)

        self.testloader = DataLoader(self.testset, batch_size=10, shuffle=False)



