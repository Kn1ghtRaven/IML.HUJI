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
import numpy

def getPenalty(penaltyString):
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
        numpyDays = numpy.array(list(penaltyDictionary.keys())) - numDays
        arg = numpy.amin(numpyDays[numpyDays > 0]) + numDays
        penaltyCategory, penaltyCost = penaltyDictionary[arg]
        penaltyCost = float(penaltyCost)
        if penaltyCategory == PENALTY_PER_NIGHT:
            factor = (penaltyCost / bookingLength)
        else:
            factor = (penaltyCost / 100)
        return factor * price

    return calculatePenalty


def seriesPenalty(penaltyStringSerie, numDays, priceList, bookingLengthList):
    returnSerie = list()
    for lineIndex in range(penaltyStringSerie.size):
        returnSerie.append(getPenalty(penaltyStringSerie[lineIndex])(numDays, priceList[lineIndex], bookingLengthList[lineIndex]))
    return pd.Series(returnSerie)

def load_test_set(filename: str):
    test_data = pd.read_csv(filename)
    return test_data


def load_data_test(filename: str):
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[["original_payment_method",
                          "hotel_star_rating",
                          "no_of_adults", "no_of_children", "no_of_extra_bed",
                          "no_of_room", "origin_country_code", "original_selling_amount"]]
    para_list = ["origin_country_code", "original_payment_method"]
    features = categorail_var(features, para_list)
    full_data[['checkin_date', 'checkout_date', "cancellation_datetime"]] = full_data[['checkin_date', 'checkout_date', "cancellation_datetime"]].apply(pd.to_datetime)
    full_data['days'] = (full_data["checkout_date"] - full_data["checkin_date"]).dt.days
    features["penalty"] = seriesPenalty(full_data['cancellation_policy_code'],11,full_data['original_selling_amount'],full_data['days'])
    labels = full_data["cancellation_datetime"]
    labels = labels.fillna(0)
    labels[labels != 0] = 1
    return features, labels

def load_data(filename: str):
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data[["original_payment_method",
                          "hotel_star_rating",
                          "no_of_adults", "no_of_children", "no_of_extra_bed",
                          "no_of_room", "origin_country_code", "original_selling_amount"]]
    para_list = ["origin_country_code", "original_payment_method"]
    features = categorail_var(features, para_list)
    full_data[['checkin_date', 'checkout_date']] = full_data[['checkin_date', 'checkout_date']].apply(pd.to_datetime)
    full_data['days'] = (full_data["checkout_date"] - full_data["checkin_date"]).dt.days
    features["penalty"] = seriesPenalty(full_data['cancellation_policy_code'],11,full_data['original_selling_amount'],full_data['days'])
    return features

# def load_data(filename: str):
#     """
#     Load Agoda booking cancellation dataset
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#
#     Returns
#     -------
#     Design matrix and response vector in either of the following formats:
#     1) Single dataframe with last column representing the response
#     2) Tuple of pandas.DataFrame and Series
#     3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
#     """
#     # TODO - replace below code with any desired preprocessing
#     full_data = pd.read_csv(filename).drop_duplicates()
#     features = full_data[["booking_datetime", "checkin_date", "checkout_date",
#                            "hotel_country_code", "hotel_star_rating",
#                           "accommadation_type_name", "charge_option", "customer_nationality",
#                           "guest_is_not_the_customer", "guest_nationality_country_name",
#                           "no_of_adults", "no_of_children", "no_of_extra_bed",
#                           "no_of_room", "origin_country_code", "original_selling_amount",
#                            "is_user_logged_in",
#                           # "cancellation_policy_code",
#                           "is_first_booking",
#                           "request_nonesmoke", "request_latecheckin",
#                           "request_highfloor", "request_largebed", "request_twinbeds",
#                           "request_airport", "request_earlycheckin"]]
#
#     features.dropna()
#     features[['checkin_date', 'checkout_date', 'booking_datetime']] = features[['checkin_date', 'checkout_date', 'booking_datetime']].apply(pd.to_datetime)
#     features['days'] = (features['checkout_date'] - features['checkin_date']).dt.days
#     features['days_in_advance'] = (features['checkin_date'] - features['booking_datetime']).dt.days
#     # features["penalty"] = getPenalty(features['cancellation_policy_code'])(11,features['original_selling_amount'],features['days'])
#     # features['charge_option'].replace(['Pay Later', 'Pay Now'], [0, 1], inplace=True)
#     # features['is_first_booking'].replace(['False', 'True'], [0, 1], inplace=True)
#     # features['is_user_logged_in'].replace(['False', 'True'], [0, 1], inplace=True)
#     # features = categorail_var(features, "customer_nationality")
#     para_list = ["customer_nationality", "accommadation_type_name", "hotel_country_code", "guest_nationality_country_name", "charge_option", "is_first_booking", "is_user_logged_in", "origin_country_code"]
#     features = features.drop(columns=["checkin_date", "booking_datetime", "checkout_date"])
#     features = categorail_var(features, para_list)
#     features = features.fillna(0)
#     labels = full_data["cancellation_datetime"]
#     labels = labels.fillna(0)
#     labels[labels != 0] = 1
#     return features, labels

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
    test = load_data("../challenge/testsetweekly/test_set_week_2.csv")
    # Fit model over data
    cols = train_X.columns.intersection(test.columns)
    estimator = AgodaCancellationEstimator().fit(train_X[cols], train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "318636081_324190693_208543520.csv")
    evaluate_and_export(estimator, test[cols], "318636081_324190693_208543520.csv")
    print(estimator.present(test_X[cols], test_y.astype('bool')))
