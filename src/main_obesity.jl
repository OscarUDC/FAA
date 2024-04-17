using Statistics
using Flux
using Flux.Losses
using FileIO;
using DelimitedFiles;
using Statistics;

# Charge the modules
include.([
    "01_oneHotEncoding.jl",
    "02_normalization.jl",
    "03_classification.jl",
    "04_accuracy.jl",
    "05_neuralnetwork_construction.jl",
    "06_neuralnetwork_training.jl",
    "07_data_parsing.jl"
])

# Cargar datos desde un archivo CSV (o el formato que estés utilizando)
data = readdlm("db\\raw\\ObesityDataSet_raw_and_data_sinthetic.csv", ',', skipstart=1) # Evita que se lea la primera línea, donde aparece el nombre de la columna

# El Genero, Columna 1, aunque ponga en el DB que la variable es categorica, no hay representaciones de más categorias que HOMBRE O MUJER así que la consideraremos binaria (Si existe el genero diamante que llamen al pentagono)
BINARY = [1, 5, 6, 10, 12]
CATEGORICAL = [9, 15, 16]

INTEGER = [7, 14]
CONTINUOUS = [2, 3 , 4, 8, 11, 13]

# Lista para almacenar todas las características transformadas
all_features = []

# Caracteristicas BINARY y CATEGORICAL
for num_col in vcat(BINARY, CATEGORICAL)
    features = [row[num_col] for row in eachrow(data)]
    classes = unique(features)
    encoded_matrix = oneHotEncoding(features, classes)
    
    # println(encoded_matrix)
    push!(all_features, encoded_matrix)
end

# Normalizar datos para las características INTEGER y CONTINUOUS
for num_col in vcat(INTEGER, CONTINUOUS)
    feature_numeric = [row[num_col] for row in eachrow(data)]
    min_max_params = calculateMinMaxNormalizationParameters(reshape(feature_numeric, :, 1))
    normalizeMinMax!(reshape(feature_numeric, :, 1), min_max_params)

    # println(feature_numeric)
    push!(all_features, feature_numeric)
end

#07_data_parsing
using Random

# Definir la proporción para los conjuntos de entrenamiento, validación y prueba
P_train = 0.7
P_val = 0.2
P_test = 0.1

# Convertir la lista de características en una matriz
all_features_matrix = hcat(all_features...)

# Obtener el tamaño total del conjunto de datos
N_total = size(data, 1)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_indices, val_indices, test_indices = holdOut(N_total, P_val, P_test)

# Separar las características para cada conjunto
data_train = all_features_matrix[train_indices, :]
data_val = all_features_matrix[val_indices, :]
data_test = all_features_matrix[test_indices, :]

# Imprimir los tamaños de los conjuntos de datos
println("Tamaño del conjunto de entrenamiento: ", size(data_train))
println("Tamaño del conjunto de validación: ", size(data_val))
println("Tamaño del conjunto de prueba: ", size(data_test))

#!Guardar la base de datos con los datos normalizados en un csv
using CSV
using DataFrames

# Convertir las características en un DataFrame
all_features_df = DataFrame(hcat(all_features...), :auto)

# Guardar el DataFrame en un archivo CSV
CSV.write("all_features.csv", all_features_df)

# Convertir las matrices de datos en DataFrames
data_train_df = DataFrame(data_train, :auto)
data_val_df = DataFrame(data_val, :auto)
data_test_df = DataFrame(data_test, :auto)

# Guardar los DataFrames en archivos CSV
CSV.write("data_train.csv", data_train_df)
CSV.write("data_val.csv", data_val_df)
CSV.write("data_test.csv", data_test_df)