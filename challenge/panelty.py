import numpy as np

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

if __name__ == "__main__":
    pen = getPenalty("45D100P_90D1N_100P")
    print(pen(50,100,100))