import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV en un DataFrame
data_path = "1Dry_Bean_Dataset.csv"  # Cambia esta ruta a tu ubicación real
df = pd.read_csv(data_path)

# Mostrar las primeras filas del DataFrame
print(df.head())



# Eliminar columnas no numéricas (si no son necesarias para el análisis)
df_cleaned = df.drop(columns='Class')

# Calcular la matriz de correlación
correlation_matrix = df_cleaned.corr()
print(correlation_matrix)

# Visualizar la matriz de correlación usando un mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')
plt.show()