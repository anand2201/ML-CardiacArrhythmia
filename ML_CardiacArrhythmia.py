import pandas
data_df = pandas.read_csv('arrhythmia.data')

a = []

for x in range(len(data_df.columns)):
    a.append(data_df.iloc[x][279])
print(sorted(a))