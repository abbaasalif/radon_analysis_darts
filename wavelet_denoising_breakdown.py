import pandas as pd
import numpy as np

df = pd.read_csv('wavelet_info.csv')
print(df.head())
# remove the follwing devices from the dataframe
df['Device Name'] = df['Device Name'].astype(str)
devices = ['1', '2', '5', '9', '17', '23', '27', '28', '36', '44', '48']
# remove the devices from the dataframe
df = df[~df['Device Name'].isin(devices)]
# wavelet family is the Wavelet column removed numbers in the string
df['Wavelet_Family'] = df['Wavelet'].str.replace('\d+', '')
df['Wavelet_Order'] = df['Wavelet'].str.extract('(\d+)')
df.fillna(0, inplace=True)
print(df.head())
# get unique counts of wavlet families
print(df['Wavelet_Family'].value_counts())
#create a new dataframe with a column called devices with all device Names with same wavelet family
df2 = df.groupby('Wavelet_Family')['Device Name'].apply(list).reset_index(name='devices')
print(df2.head())
# save df2 to a csv file
df2.to_csv('wavelet_devices_families.csv', index=False)