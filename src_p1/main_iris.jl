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
data = readdlm("db\\raw\\iris.data", ',')

# Extracción de características y clases
features = [row[end] for row in eachrow(data)]
classes = unique(features)

# Llamada a la función oneHotEncoding
encoded_matrix = oneHotEncoding(features, classes)

# Imprimir la matriz codificada
println("Matriz Codificada:")
println(encoded_matrix)

# Calcular parámetros para Min-Max Normalization
min_max_params = calculateMinMaxNormalizationParameters(encoded_matrix)

# Normalizar entre 0 y 1 (modificando el array original)
normalizeMinMax!(encoded_matrix, min_max_params)
println("\nMatriz Normalizada (0-1) con parámetros dados:")
println(encoded_matrix)

# Crear nueva matriz y normalizar entre 0 y 1 (sin modificar el array original)
normalized_matrix_0_1 = normalizeMinMax(encoded_matrix, min_max_params)
println("\nNueva Matriz Normalizada (0-1) con parámetros dados:")
println(normalized_matrix_0_1)

# Convertir la matriz a tipo Float64 antes de normalizar a media 0
encoded_matrix = Float32.(encoded_matrix)

# Calcular parámetros para Zero Mean Normalization
zero_mean_params = calculateZeroMeanNormalizationParameters(encoded_matrix)

# Normalizar a media 0 (modificando el array original)
normalizeZeroMean!(encoded_matrix, zero_mean_params)
println("\nMatriz Normalizada con Media 0 con parámetros dados:")
println(encoded_matrix)

# Crear nueva matriz y normalizar a media 0 (sin modificar el array original)
normalized_matrix_zero_mean = normalizeZeroMean(encoded_matrix, zero_mean_params)
println("\nNueva Matriz Normalizada con Media 0 con parámetros dados:")
println(normalized_matrix_zero_mean)

# Crear una nueva ANN
ann = buildClassANN(4, [4, 3, 4], 3)

println("\nSe ha conseguido crear la red neuronal")
println(ann.layers)

ann = buildClassANN(3,[2, 2, 2], 2)

println("\nSe ha conseguido crear la red neuronal")
println(ann.layers)

#holdOut

tupla = holdOut(100, 0.15)
println()
println(isa(tupla, Tuple))
println(length(tupla[1])/100, "\n", length(tupla[2])/100)

tupla = holdOut(100, 0.15, 0.15)
println()
println(isa(tupla, Tuple))
println(length(tupla[1])/100, "\n", length(tupla[2])/100, "\n", length(tupla[3])/100)

# confusionMatrix

# Test 1: Matrices booleanas multiclase
outputs_bool = rand(Bool, 100, 5)
targets_bool = rand(Bool, 100, 5)

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_bool, targets_bool)

# Test 2: Matrices reales multiclase
outputs_real = rand(100, 5)
targets_bool = rand(Bool, 100, 5)

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_real, targets_bool)

# Test 3: Vectores de tipo Any multiclase
outputs_any = ["class1", "class2", "class3", "class1", "class2", "class3"]
targets_any = ["class1", "class2", "class3", "class1", "class2", "class3"]

# Imprimir las matrices de confusión
printConfusionMatrix(outputs_any, targets_any)




# Matriz de salidas balanceada
outputs_balanced = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 0 0 1 0;
    0 0 0 0 1
]

# Matriz de salidas desbalanceada
outputs_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Matriz de salidas aleatorias (seleccionando una clase más probable)
outputs_random = [
    1 0 0 0 0;
    0 1 0 0 0;
    0 0 1 0 0;
    0 1 0 0 0;
    0 0 0 0 1
]

# Matriz de salidas aleatorias desbalanceada (seleccionando una clase menos probable)
outputs_random_imbalanced = [
    1 0 0 0 0;
    1 0 0 0 0;
    1 0 0 0 0;
    2 0 0 0 0;
    1 0 0 0 0
]

# Test 1: Matrices booleanas multiclase
targets_bool = rand(Bool, 5, 5)

# Imprimir las matrices de confusión sin ponderación
printConfusionMatrix(outputs_balanced, targets_bool, weighted=false)

# Imprimir las matrices de confusión con ponderación
printConfusionMatrix(outputs_balanced, targets_bool, weighted=true)