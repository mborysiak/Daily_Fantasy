#%%

import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('c:/Users/mborysia/Desktop/DK_Contests.xlsx')

# df['IsDoubleUp'] = 0
# df.loc[df.Name.str.contains('Double Up'), 'IsDoubleUp'] = 1
df = df[~df.Name.str.contains('Double Up')].reset_index(drop=True)

week_cols =  pd.get_dummies(df.Week, prefix='Week', prefix_sep='_', drop_first=True)
df = pd.concat([df, week_cols], axis=1)
df['Pool_Per_Entry'] = df.PrizePool / df.Entries
df = df.dropna().reset_index(drop=True)

# cols.extend(week_cols)

# X = df[cols]
# X_sc = StandardScaler().fit_transform(X)
# y = df[['CashLine']]

cols = ['BuyIn', 'TopPrize', 'Entries', 'MaxEntries']

sc = StandardScaler()
sc.fit(df[cols])
df_sc = pd.DataFrame(sc.transform(df[cols]), columns=cols)
df_sc = pd.concat([df_sc, df[['CashLine']]], axis=1)

import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm("CashLine ~ BuyIn + TopPrize + Entries + MaxEntries", 
                 df_sc, groups=df["Week"])
mdf = md.fit()
print(mdf.summary())
#%%

X_test = pd.DataFrame([[33, 25000, 7067, 5]], columns=cols)
X_test = sc.transform(X_test)
X_test = sm.add_constant(X_test)
#%%
mdf.predict(exog=[-0.27, -0.03, -0.16, -0.364])

# %%
