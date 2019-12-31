"""
Contains the class Ensemble_model for the ensemble models management.
"""
import pandas as pd
import numpy as np
from pandarallel import pandarallel

# These values are determined by heuristics
P_fraud       = 0.006         # prior fraud probability.
P_noFraud     = 1 - P_fraud   # prior true transaction probability.
C_FN          = 30            # cost of false negative.
C_FP          = 1             # cost of false positive.
# the following  2 are constant cause we know the proportions of the training sets we build:
#n_P           = 383
#n_N           = 2000 - n_P
 
def ECM(n_FN, n_FP, n_P, n_N):
    """
    Expected Cost of Misclassification:
    ECM = C_FN * P(Fraud) * n_FN / n_P + C_FP * P(Non-Fraud) * (n_FP / n_N)
    """
    frst   = C_FN * P_fraud   * n_FN / n_P
    scnd   = C_FP * P_noFraud * n_FP / n_N
    return frst + scnd

class Ensemble_model:
    def __init__(self, models, val_df, cutoff):
        # Here the parameter 'val_df' could be the validation dataframe, but also the instaces we want to evaluate (to predict probability of fraud).
        self.error_df                  = pd.DataFrame()
        self.error_df['labels']        = np.array(val_df.Class)
        self.error_df['probability']   = 0
        
        
        for model in models: 
            self.error_df['probability']   += model.predict_proba(val_df.drop(['Class'], axis=1))[:,1] / len(models)  # We do this as a way of soft voting. The probability of fraud for the instances for each model is averaged.
            
        
        self.error_df['labels_bool']   = self.error_df.labels != 0

        # To make predictions we call error_df[['probablility','predictions']]
        self.error_df['predictions']   = self.error_df.probability > cutoff
        self.error_df['FN']            = self.error_df.parallel_apply(lambda row: True if (row.labels_bool != row.predictions) and (row.predictions == False) else False, axis=1)
        self.error_df['FP']            = self.error_df.parallel_apply(lambda row: True if (row.labels_bool != row.predictions) and (row.predictions == True) else False, axis=1)
        self.error_df['TP']            = self.error_df.parallel_apply(lambda row: True if (row.labels_bool == row.predictions) and (row.predictions == True) else False, axis=1)
        self.error_df['TN']            = self.error_df.parallel_apply(lambda row: True if (row.labels_bool == row.predictions) and (row.predictions == False) else False, axis=1)

        self.FNs                       = self.error_df.FN.sum()
        self.FPs                       = self.error_df.FP.sum()
        self.TPs                       = self.error_df.TP.sum() 
        self.sample_size               = len(self.error_df)
        self.sensitivity               = self.error_df.TP.sum() / (self.error_df.FN.sum() + self.error_df.TP.sum()) # The proportion of actual positives that are correctly classified as such.
        self.specificity               = self.error_df.TN.sum() / (self.error_df.FP.sum() + self.error_df.TN.sum()) # The proportion of actual negatives that are correctly classified as such.
        
        self.n_P                            = val_df.Class.sum() # number of fraud on test set
        self.n_N                            = len(val_df) - self.n_P  # number of true transactions on test set
        self.ECM                       = ECM(self.FNs, self.FPs, self.n_P, self.n_N)  
              
    # A brute force approach to find the threshold that renders min(ECM)
    # TO DO: parallelize this, or find an ellegant wat to minimize the ECM
    def find_min_ECM(self):
        """
        Returns the cutoff value
        """
        ecm_list = []
        for n in range(0,10):
            p           = n * 0.1
            check_error = pd.DataFrame()

            check_error['predictions'] = self.error_df.probability > p
            check_error['labels_bool'] = self.error_df.labels_bool

            check_error['FN']          = check_error.parallel_apply(lambda row: True if (row.labels_bool != row.predictions) and (row.predictions == False) else False, axis=1)
            check_error['FP']          = check_error.parallel_apply(lambda row: True if (row.labels_bool != row.predictions) and (row.predictions == True) else False, axis=1)
            check_error['TP']          = check_error.parallel_apply(lambda row: True if (row.labels_bool == row.predictions) and (row.predictions == True) else False, axis=1)

            fn = check_error.FN.sum()
            fp = check_error.FP.sum()
            ecm_list.append(ECM(fn, fp, self.n_P, self.n_N))

        self.min_ECM  =  np.argmin(ecm_list) * 0.1
        return           np.argmin(ecm_list) * 0.1

    def summary(self):
        """
        Prints a summary with the metrics of the model.
        """
        print("False negatives: \t" + str(self.FNs))
        print("False positives: \t" + str(self.FPs))
        print("True positives: \t"  + str(self.TPs))
        print("Sample size: \t\t"   + str(self.sample_size))
        print("Sensitivity: \t\t"   + str(self.sensitivity))
        print("Specificity: \t\t"   + str(self.specificity))
        print("ECM: \t\t\t"         + str(self.ECM) + "\n")
   