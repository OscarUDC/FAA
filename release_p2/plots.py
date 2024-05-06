import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv("db\\raw\\ObesityDataSet_raw_and_data_sinthetic.csv")

# Eliminar la columna de índice si es necesaria
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# Crear el gráfico de pares con un tamaño de cuadro más grande
pair_plot = sns.pairplot(data, hue="NObeyesdad", height=3, aspect=1.5)  # Ajusta el valor de height y aspect según tu preferencia

# Guardar el gráfico de pares como una imagen
pair_plot.savefig("pair_plot.png")
plt.close()  # Cerramos el gráfico para liberar memoria

# Histograma de todas las columnas numéricas
data.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# Contar las ocurrencias de cada valor en las columnas categóricas
for column in data.select_dtypes(include='object'):
    plt.figure(figsize=(10, 6))
    sns.countplot(data[column])
    plt.title(f'Distribución de {column}')
    plt.xticks(rotation=45)
    plt.savefig(f"distribution_{column}.png")
    plt.close()

# Crear gráficos de dispersión para las columnas numéricas
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
scatter_plot = sns.pairplot(data[numeric_columns])
scatter_plot.savefig("scatter_plot.png")
plt.close()
