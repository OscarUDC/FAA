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
    "06_neuralnetwork_training.jl"
])

# Cargar datos desde un archivo CSV (o el formato que estés utilizando)
data = readdlm("db\\iris.data", ',')

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
