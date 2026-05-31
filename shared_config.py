from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from enum import Enum
from pathlib import Path
import pickle
  
class FE(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None ):
        return self
    
    def transform(self, X):
        X = X.copy()

        # romove redundant labels 
        cat_tofix= ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',]
        for feature in cat_tofix:
            X[feature]  = X[feature].replace({1: 0, 2:1})

        X['average_charges'] = X['TotalCharges'] / (X['tenure'] + 1)
        X['monthly_to_total'] = X['MonthlyCharges'] / (X['TotalCharges']+ 1)
        X['Family'] = X['Partner_Yes'] + X['Dependents_Yes']
        X = X.drop(['gender_Male','Dependents_Yes', 'Partner_Yes', 'TotalCharges', 'MonthlyCharges', 'tenure'], axis = 1)
        return X 
    
    def set_output(self, transform= None):
        self._transform_output = transform
        return self

class GenderEnum(str, Enum):
    Female= 'Female'
    Male = 'Nale'

class ContractEnum(str, Enum):
    Month = 'Month-to-month' 
    Year =  'One year' 
    TwoYear = 'Two year'

class DependentsEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'

class DeviceProtectionEnum(str, Enum): 
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'

class InternetServiceEnum(str, Enum):
    DSL = 'DSL'
    Fabric =  'Fiber optic'
    No = 'No'

class MultipleLinesEnum(str, Enum):
    NoPhone = 'No phone service'
    Yes = 'Yes'
    No = 'No'

class OnlineBackupEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'

class OnlineSecurityEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'

class PaperlessBillingEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'

class PartnerEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'

class PaymentMethodEnum(str, Enum):
        Electronic =  'Electronic check'
        Mail = 'Mailed check'
        Bank = 'Bank transfer (automatic)' 
        Credit = 'Credit card (automatic)'

class PhoneServiceEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'



class StreamingMoviesEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'

class StreamingTVEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'


class TechSupportEnum(str, Enum):
    Yes = 'Yes'
    No = 'No'
    NoInternet = 'No internet service'
