#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
# %%
df = pd.read_csv("../data/product_performance.csv", sep=',')
# %%
# %%
df['Good_Performance'].replace(0, np.nan, inplace=True)
# %%
df.columns
# %%
columns_to_fix = ['Product_Price', 'Discount_Rate', 'Product_Rating', 'Number_of_Reviews',
       'Stock_Availability', 'Days_to_Deliver', 'Return_Rate', 'Category_ID',
       'Performance_Score']

df[columns_to_fix] = df[columns_to_fix].fillna(df[columns_to_fix].mean())
# %%
df.to_csv('../data/product_performance_cleaned.csv', index=False)
# %%
# Scale the features (excluding Good_Performance)
features = df.drop('Good_Performance', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Create final scaled dataframe
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['Good_Performance'] = df['Good_Performance'].values
# %%
scaled_df.head()
# %%
scaled_df.to_csv('../data/scaled_product_performance.csv', index=False)

# %%
