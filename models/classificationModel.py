#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
# %%
df_labeled = pd.read_csv('../data/scaled_product_performance.csv')
# %%
df_labeled
# %%
df_labeled['Good_Performance'] = df_labeled['Good_Performance'].fillna(0)
# %%
print(df_labeled['Good_Performance'].value_counts()) 
#%%
x = df_labeled.drop(['Good_Performance', 'Performance_Score'], axis=1)
y = df_labeled['Good_Performance']
# %%
#*Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
# %%
model = RandomForestClassifier(random_state=42)

# Definir os hiperparâmetros para o GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar o GridSearchCV
rf_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Realizar o treinamento com o GridSearchCV
tqdm.write("Iniciando o treino do modelo com GridSearchCV...")
rf_grid.fit(X_train, y_train)
#%%
best_rf = rf_grid.best_estimator_
tqdm.write("Iniciando a previsão...")
y_pred = best_rf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
# %%

# Agora extraímos as importâncias das features
importances = best_rf.feature_importances_

# Obtemos os nomes das features a partir de X (as variáveis independentes)
feature_names = x.columns

# Criamos um DataFrame para organizar as features e suas importâncias
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# %%
