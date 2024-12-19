import numpy as np
import pandas as pd

df = pd.read_csv('nbeats_smape.csv')

avg_smape = np.mean(df['SMAPE'])
std_dev_smape = np.std(df['SMAPE'])

print("The average sMAPE for NBeats model is:", avg_smape)
print("The Standard Deviation of sMAPE for NBeats model is:", std_dev_smape)