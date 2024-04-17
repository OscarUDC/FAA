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
data = readdlm("db\\ObesityDataSet_raw_and_data_sinthetic.csv", ',', skipstart=1) # Evita que se lea la primera línea, donde aparece el nombre de la columna

CATEGORICAL = [1, 9, 15, 16]
BINARY = [5, 6, 10, 12]

INTEGER = [7, 14]
CONTINUOUS = [2, 3 , 4, 8, 11, 13]

# Lista para almacenar todas las características transformadas
all_features = []

# Caracteristicas CATEGORICAL
for num_col in CATEGORICAL
    features = [row[num_col] for row in eachrow(data)]
    classes = unique(features)
    encoded_matrix = oneHotEncoding(features, classes)

    # Imprimir la matriz codificada (clases)
    println("Matriz Codificada para la columna $num_col:")
    println(encoded_matrix)
end

# Caracteristicas BINARY
for num_col in BINARY
    features = [row[num_col] for row in eachrow(data)]
    classes = unique(features)
    encoded_matrix = oneHotEncoding(features, classes)

    # Imprimir la matriz codificada (clases)
    println("Matriz Codificada para la columna $num_col:")
    println(encoded_matrix)
end

# Normalizar datos para las caracteristicas entegras y continuas
# Caracteristicas INTEGER
for num_col in INTEGER
    # Obtener los valores de la columna 7
    feature_numeric = [row[num_col] for row in eachrow(data)]

    # Calcular los parámetros de normalización
    min_max_params = calculateMinMaxNormalizationParameters(reshape(feature_numeric, :, 1))

    # Imprimir los parámetros de normalización
    println("Parámetros de Normalización $num_col:")
    println("Mínimo: ", min_max_params[1])
    println("Máximo: ", min_max_params[2])

    # Normalizar los datos
    normalizeMinMax!(reshape(feature_numeric, :, 1), min_max_params)

    # Imprimir los datos normalizados
    println("Datos normalizados para la columna $num_col:")
    println(feature_numeric)
end

# Caracteristicas CONTINUOUS
for num_col in CONTINUOUS
    # Obtener los valores de la columna 7
    feature_numeric = [row[num_col] for row in eachrow(data)]

    # Calcular los parámetros de normalización
    min_max_params = calculateMinMaxNormalizationParameters(reshape(feature_numeric, :, 1))

    # Imprimir los parámetros de normalización
    println("Parámetros de Normalización $num_col:")
    println("Mínimo: ", min_max_params[1])
    println("Máximo: ", min_max_params[2])

    # Normalizar los datos
    normalizeMinMax!(reshape(feature_numeric, :, 1), min_max_params)

    # Imprimir los datos normalizados
    println("Datos normalizados para la columna $num_col:")
    println(feature_numeric)
end

#07_data_parsing
using Random

# Definir la proporción para los conjuntos de entrenamiento, validación y prueba
P_train = 0.7
P_val = 0.2
P_test = 0.1

# Obtener el tamaño total del conjunto de datos
N_total = size(data, 1)

# Calcular los índices para dividir los datos
train_indices, val_indices, test_indices = holdOut(N_total, P_val, P_test)

# Separar los datos en conjuntos de entrenamiento, validación y prueba
data_train = data[train_indices, :]
data_val = data[val_indices, :]
data_test = data[test_indices, :]

# Imprimir los tamaños de los conjuntos de datos
println("Tamaño del conjunto de entrenamiento: ", size(data_train))
println("Tamaño del conjunto de validación: ", size(data_val))
println("Tamaño del conjunto de prueba: ", size(data_test))