from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd

PENALTY_DELIMITER = "_"
NUMBER_OF_DAYS_DELIMITER = "D"
PENALTY_PER_NIGHT = "N"
PENALTY_PERCENTILE = "P"
import re

def getPenalty(penaltyString):
    def func(a,b,c):
        return 0
    if penaltyString == "UNKNOWN":
        return func
    penaltyTypes = penaltyString.split("_")
    penaltyDictionary = dict()
    for clause in penaltyTypes:
        penaltyCategory = ""
        penaltyCost = -1

        #extract the penalty and the number of days within
        #the cancelation need to be made
        if re.search(NUMBER_OF_DAYS_DELIMITER, clause):
            numDays, penalty = re.split(NUMBER_OF_DAYS_DELIMITER, clause)
        else:
            numDays = 0
            penalty = clause

        # extract the penalty category and the penalty factor
        if re.search(PENALTY_PER_NIGHT, penalty):
            penaltyCategory = PENALTY_PER_NIGHT
            penaltyCost = int(penalty[:-1])
        elif re.search(PENALTY_PERCENTILE, penalty):
            penaltyCategory = PENALTY_PERCENTILE
            penaltyCost = int(penalty[:-1])

        #insertion of the penalty into our penaltyDictionary
        if (penaltyCategory != "") & (penaltyCost != -1):
            penaltyDictionary[int(numDays)] = (penaltyCategory, penaltyCost)

    #The function that will calculate the cancelation penalty
    def calculatePenalty(numDays, price, bookingLength):
        numpyDays = np.array(list(penaltyDictionary.keys())) - numDays
        if not any(numpyDays[numpyDays > 0]):
            arg = np.amin(numpyDays) + numDays
        else:
            arg = np.amin(numpyDays[numpyDays > 0]) + numDays
        penaltyCategory, penaltyCost = penaltyDictionary[arg]
        penaltyCost = float(penaltyCost)
        if penaltyCategory == PENALTY_PER_NIGHT:
            factor = (penaltyCost / bookingLength)
        else:
            factor = (penaltyCost / 100)
        return factor * price

    return calculatePenalty


def seriesPenalty(penaltyStringSerie, numDays, priceList, bookingLengthList):
    priceList = priceList.to_numpy()
    bookingLengthList = bookingLengthList.to_numpy()
    returnSerie = list()
    for lineIndex in range(penaltyStringSerie.size):
        func = getPenalty(penaltyStringSerie[lineIndex])
        price = priceList[lineIndex]
        bookingdays = bookingLengthList[lineIndex]
        res = func(numDays, price, bookingdays)
        returnSerie.append(res)
    return pd.Series(returnSerie)

def load_test_set(filename: str):
    test_data = pd.read_csv(filename)
    special_requests = ['request_nonesmoke', 'request_latecheckin',
                        'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport',
                        'request_earlycheckin']
    test_data[['checkin_date', 'checkout_date', 'booking_datetime']] = test_data[['checkin_date', 'checkout_date', 'booking_datetime']].apply(pd.to_datetime)
    test_data['days'] = (test_data['checkout_date'] - test_data['checkin_date']).dt.days
    test_data['days_in_advance'] = (test_data['checkin_date'] - test_data['booking_datetime']).dt.days
    for name in special_requests:
        test_data[name] = test_data[name].fillna(0)
    test_data['total_of_special_requests'] = sum([test_data[col] for col in special_requests])
    # for the penaltys
    test_data[['checkin_date', 'checkout_date']] = test_data[['checkin_date', 'checkout_date']].apply(pd.to_datetime)
    test_data['days'] = (test_data["checkout_date"] - test_data["checkin_date"]).dt.days
    test_data["penaltys"] = seriesPenalty(test_data['cancellation_policy_code'], 11, test_data['original_selling_amount'], test_data['days'])
    test_data['days_in_advance'] = (test_data['checkin_date'] - test_data['booking_datetime']).dt.days
    return test_data


def load_data_test(filename: str):
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[["original_payment_method",
                          "hotel_star_rating",
                          "no_of_adults", "no_of_children", "no_of_extra_bed",
                          "no_of_room", "origin_country_code", "original_selling_amount",'request_nonesmoke',
                          'request_latecheckin',
                          'request_highfloor', 'request_largebed',
                          'request_twinbeds', 'request_airport',
                          'request_earlycheckin', "cancellation_datetime", 'checkin_date', 'checkout_date', 'booking_datetime'
                        ,'cancellation_policy_code']]
    para_list = ["origin_country_code", "original_payment_method"]
    features = categorail_var(features, para_list)
    features[['checkin_date', 'checkout_date', "cancellation_datetime"]] = features[['checkin_date', 'checkout_date', "cancellation_datetime"]].apply(pd.to_datetime)
    features['days'] = (features["checkout_date"] - features["checkin_date"]).dt.days
    features["penaltys"] = seriesPenalty(features['cancellation_policy_code'], 11, features['original_selling_amount'], features['days'])
    features['cancellation_time'] = features['cancellation_datetime'].fillna(features['checkin_date'])  # need to see if this line works
    features[['checkin_date', 'checkout_date', 'booking_datetime', 'cancellation_time']] = features[['checkin_date', 'checkout_date', 'booking_datetime','cancellation_time']].apply(pd.to_datetime)
    features['time_to_cancel'] = (features['cancellation_time'] - features['booking_datetime']).dt.days  # need to see if it is possitive or negative
    # features = features[features['time_to_cancel'] < 12]# need to recheck this line
    special_requests = ['request_nonesmoke', 'request_latecheckin',
                        'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport',
                        'request_earlycheckin']
    no_null = [ 'hotel_star_rating', 'no_of_adults',
            'no_of_children',           'no_of_extra_bed',
                'no_of_room',   'original_selling_amount']
    for name in no_null:
        features[name] = features[name].fillna(0)
    for name in special_requests:
        features[name] = features[name].fillna(0)
    features['total_of_special_requests'] = sum(
        [features[col] for col in special_requests])
    features["cancellation_datetime"] = features["cancellation_datetime"].fillna(0)
    # features["cancellation_datetime"][features["cancellation_datetime"] != 0] = 1
    features.loc[features['cancellation_datetime'] != 0, 'cancellation_datetime'] = 1
    features['days'] = (features['checkout_date'] - features['checkin_date']).dt.days
    features['days_in_advance'] = (features['checkin_date'] - features['booking_datetime']).dt.days
    labels = features["cancellation_datetime"]
    features = features.drop(
            columns=["cancellation_datetime", 'checkin_date', 'checkout_date', 'booking_datetime', 'cancellation_policy_code'])
    features = features.drop(
        columns=no_null)
    return features, labels

def categorail_var(features , str_var_list):
    for str_var in str_var_list:
        var = features[str_var].to_list()
        pd_var = pd.Series(var)
        df_var = pd.get_dummies(pd_var)
        features = pd.concat([features, df_var], axis='columns')
    features = features.drop(columns=str_var_list)
    return features


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data_test("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    test = load_test_set("../challenge/testsetweekly/test_set_week_4.csv")
    # Fit model over data
    cols = train_X.columns.intersection(test.columns)
    estimator = AgodaCancellationEstimator(train_X[cols]).fit(train_X[cols], train_y)
    # estimator.report(test_X[cols], test_y)
    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "318636081_324190693_208543520.csv")
    evaluate_and_export(estimator, test[cols], "318636081_324190693_208543520.csv")
    print(estimator.present(test_X[cols], test_y.astype('bool')))
