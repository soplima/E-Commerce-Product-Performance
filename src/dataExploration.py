#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/product_performance.csv", sep=',')
# %%
print('Head of the dataset')
df.head(10) 

#!apply log transformation on Product_Price and Number pf reviews
# %%
print('Data info:')
df.info()
# %%
print('Count null values:')
df.isnull().sum()
# %%
print('Descriptive statistics: ')
df.describe()
# %%
df['Performance_Score'] = (
    df['Product_Rating'] * 0.4 +
    np.log1p(df['Number_of_Reviews']) * 0.3 +
    (1 - df['Return_Rate']) * 0.2 +
    df['Stock_Availability'] * 0.1
)
#%%
df['Performance_Score'].describe()
#%%
#!Plot distribution for Performance_Score
plt.figure(figsize=(10, 6))
sb.histplot(df['Performance_Score'], kde=True, bins=30, color='skyblue')
plt.axvline(df['Performance_Score'].quantile(0.75), color='red', linestyle='--', 
            label=f'75th Percentile: {df["Performance_Score"].quantile(0.75):.2f}')
plt.title('Distribution of Performance Score')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#%%
df['Good_Performance'] = (df['Performance_Score'] >= 3.64).astype(int) 
df
df['Good_Performance'].value_counts()
#%%
#!Plot correlation for each feature
correlation_matrix = df.drop(columns=['Performance_Score']).corr()
plt.figure(figsize=(8, 6))
sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Performance Score')
plt.tight_layout()
plt.show() #higher correlation between good performance, product rating and number of reviews
# %%
#!Plot distribution for each feature
plt.figure(figsize=(15, 12))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sb.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel("")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# %%
#!Normalize Number of Reviews
df['Number_of_Reviews'] = np.log1p(df['Number_of_Reviews'])
# %%
df['Number_of_Reviews']
# %%
df.to_csv('../data/product_performance.csv', index=False)

# %%
