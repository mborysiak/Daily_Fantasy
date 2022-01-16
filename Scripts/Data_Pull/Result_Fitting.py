#%%
import pandas as pd
import os
import zipfile
import numpy as np
download_path = 'c:/Users/mborysia/Downloads/'
extract_path = download_path + 'DK_Results/'

#%%
zip_files = [f for f in os.listdir(download_path) if '.zip' in f]

for f in zip_files:
    with zipfile.ZipFile(f'{download_path}/{f}',"r") as zip_ref:
        zip_ref.extractall(f'{extract_path}/')
    
    os.replace(f'{download_path}/{f}', f'{extract_path}/Archive/{f}')

#%%

play_action = True
million = False

if million: (low, high) = (10000000, 70000000)
if play_action: (low, high) = (70000000, 900000000)

csv_files = [f for f in os.listdir(extract_path) if \
             os.path.getsize(extract_path+f) > low and \
                 os.path.getsize(extract_path+f) < high]

# csv_files = [f for f in os.listdir(extract_path) if \
#              os.path.getsize(extract_path+f) > 10000000 and \
#                  os.path.getsize(extract_path+f) < 70000000]

df = pd.DataFrame()
for f in csv_files:
    df = pd.concat([df, pd.read_csv(extract_path+f, low_memory=False)[['Points', 'Rank']]], axis=0)

if million:
    scores = pd.DataFrame([[1,1000000],
                            [1,125000],
                            [1,75000],
                            [1,50000],
                            [1,30000],
                            [1,20000],
                            [2,15000],
                            [2,10000],
                            [5,7000],
                            [5,5000],
                            [10,3500],
                            [10,2500],
                            [20,1500],
                            [30,1000],
                            [35,750],
                            [50,600],
                            [50,500],
                            [75,400],
                            [100,300],
                            [150,250],
                            [200,200],
                            [300,150],
                            [450,125],
                            [750,100],
                            [1250,80],
                            [3500,60],
                            [9000,40],
                            [30000,30]])

if play_action:
    scores = pd.DataFrame([[1,100000],
                            [1,50000],
                            [1,30000],
                            [1,20000],
                            [1,15000],
                            [1,10000],
                            [2,7500],
                            [2,5000],
                            [5,3000],
                            [5,2000],
                            [5,1000],
                            [10,750],
                            [10,500],
                            [25,250],
                            [30,150],
                            [50,100],
                            [100,75],
                            [250,50],
                            [500,40],
                            [750,30],
                            [1250,25],
                            [2000,20],
                            [4000,15],
                            [7000,12],
                            [10000,10],
                            [16750,8],
                            [30000,6],
                            [50000,5]])

scores.columns = ['num_finishes', 'prize']
scores['Rank'] = scores.num_finishes.cumsum()
df['prize'] = 0

for i in range(len(scores)):
    if i == 0: 
        score_i = scores.loc[i, 'prize']
        df.loc[df.Rank==1, 'prize'] = score_i
    else:
        score_i = scores.iloc[i]
        score_i_minus = scores.iloc[i-1]
        df.loc[(df.Rank > score_i_minus.Rank) & (df.Rank <= score_i.Rank), 'prize'] = score_i.prize

df = df[df.prize > 0].reset_index(drop=True)

# points = list(df.groupby('prize').agg(Points=('Points', lambda x: np.percentile(x, 50))).sort_values('Points', ascending=False)['Points'].values)
points = list(df.groupby('prize').agg({'Points': 'mean'}).sort_values('Points', ascending=False)['Points'].values)
print([np.log(p*1.02) for p in points])

#%%
import matplotlib.pyplot as plt

X_pred = df.prize.unique()
plt.scatter(np.log(df.prize), df.Points)
plt.plot(np.log(X_pred), points,  'r', lw=3)
plt.show()

# %%

# from scipy.interpolate import UnivariateSpline
# from sklearn.linear_model import LinearRegression

# low_results = df[df.prize < 0.4]
# low_results = low_results.sample(frac=0.1)
# high_results = df[df.prize > 0.4]
# train = pd.concat([high_results, low_results], axis=0).reset_index(drop=True)
# train = train.sort_values(by='Rank').reset_index(drop=True)

# X = train[['prize']]
# y = train[['Points']]

# spl = UnivariateSpline(X, y, k=2, s=1000)
# lr = LinearRegression()
# lr.fit(X, y)
# print(lr.coef_, lr.intercept_)



# %%
# %%
