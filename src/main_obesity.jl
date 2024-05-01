using Statistics
using Flux
using Flux.Losses
using FileIO;
using DelimitedFiles;
using Statistics;

# Charge the modules
include.([
"soluciones.jl"
])

# Cargar datos desde un archivo CSV (o el formato que estés utilizando)
data = readdlm("db\\raw\\ObesityDataSet_raw_and_data_sinthetic.csv", ',', skipstart=1) # Evita que se lea la primera línea, donde aparece el nombre de la columna

#01_oneHotEncoding
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
    # Agregar características binarias y categóricas codificadas
    push!(all_features, encoded_matrix)
end

# Normalizar datos para las características INTEGER y CONTINUOUS
for num_col in vcat(INTEGER, CONTINUOUS)
    feature_numeric = [row[num_col] for row in eachrow(data)]
    min_max_params = calculateMinMaxNormalizationParameters(reshape(feature_numeric, :, 1))
    normalizeMinMax!(reshape(feature_numeric, :, 1), min_max_params) #! Hay que mirar si la normalización ZeroMean puede ser mejor para codificar algunas caracteristicas

    # println(feature_numeric)
    # Agregar características enteras y continuas codificadas
    push!(all_features, feature_numeric)
end

#! println("Caracteristicas + Targets", all_features)

#07_data_parsing
using Random

# Definir la proporción para los conjuntos de entrenamiento, validación y prueba
P_train = 0.7
P_val = 0.2
P_test = 0.1

# Convertir la lista de características en una matriz
all_features_matrix = hcat(all_features...) #! El problema está que al convertir a matriz las columnas categoricas se hacen varias columnas en vez de que sean una sola
#! println("Matriz", all_features_matrix)

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


#09_cross_validation
# Importamos CUDA para ejecutar el programa en la GPU
# using Pkg
# Pkg.add("CUDA")
# Pkg.add("cuDNN")
using CUDA

# Entrenamiento
# Obtener los objetivos del conjunto de entrenamiento
targets_train = data_train[:, 8] # Los targets están en la posicion 8 de la lista porque son los octavos en hacer el OneHotEncoding
inputs_train = data_train[:, [1:7; 9:end]]

# Imprimir los objetivos del conjunto de entrenamiento
println("Objetivos del conjunto de entrenamiento: ", targets_train)
println("Objetivos del conjunto de entrenamiento: ", inputs_train)


# Define los hiperparámetros del modelo (ajústalos según tus necesidades)
modelHyperparameters = Dict(
    "topology" => [10, 5, 1],  # Ejemplo de topología de red neuronal
    "learningRate" => 0.01,
    "maxEpochs" => 1000
)

# Define el número de folds para la validación cruzada
num_folds = 5

# Genera los índices de validación cruzada
crossValidationIndices = crossvalidation(targets_train, num_folds)

# Aplica modelCrossValidation en la GPU
mean_accuracy, std_accuracy, mean_error_rate, std_error_rate, mean_recall, std_recall,
mean_specificity, std_specificity, mean_precision, std_precision, mean_npv, std_npv,
mean_f1, std_f1 = gpu(modelCrossValidation)(:ANN, modelHyperparameters, inputs_train, targets_train, crossValidationIndices)

# Imprime los resultados
println("Accuracy: ", mean_accuracy, " ± ", std_accuracy)
println("Error rate: ", mean_error_rate, " ± ", std_error_rate)
println("Recall: ", mean_recall, " ± ", std_recall)
println("Specificity: ", mean_specificity, " ± ", std_specificity)
println("Precision: ", mean_precision, " ± ", std_precision)
println("NPV: ", mean_npv, " ± ", std_npv)
println("F1 Score: ", mean_f1, " ± ", std_f1)