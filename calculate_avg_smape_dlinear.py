import numpy as np
import pandas as pd

df = pd.read_csv('dlinear_smape.csv')

avg_smape = np.mean(df['SMAPE'])
std_dev_smape = np.std(df['SMAPE'])

print("The average sMAPE for dlinear model is:", avg_smape)
print("The Standard Deviation of sMAPE for dlinear model is:", std_dev_smape)