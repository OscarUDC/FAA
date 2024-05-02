using Statistics
using Flux
using Flux.Losses
using FileIO;
using DelimitedFiles;
using Statistics;
using Random;
using CUDA;
using DataFrames;
using CSV;

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


TARGETS = 17
# Lista para almacenar todas las características transformadas
inputs_train = []
targets_train = []

# Caracteristicas BINARY y CATEGORICAL
for num_col in vcat(BINARY, CATEGORICAL)
    features = [row[num_col] for row in eachrow(data)]
    classes = unique(features)
    
    encoded_matrix = oneHotEncoding(features, classes)

    # println(encoded_matrix)
    # Agregar características binarias y categóricas codificadas
    push!(inputs_train, encoded_matrix)
end

# Normalizar datos para las características INTEGER y CONTINUOUS
for num_col in vcat(INTEGER, CONTINUOUS)
    feature_numeric = [row[num_col] for row in eachrow(data)]
    min_max_params = calculateMinMaxNormalizationParameters(reshape(feature_numeric, :, 1))
    normalizeMinMax!(reshape(feature_numeric, :, 1), min_max_params) #! Hay que mirar si la normalización ZeroMean puede ser mejor para codificar algunas caracteristicas

    # println(feature_numeric)
    # Agregar características enteras y continuas codificadas
    push!(inputs_train, feature_numeric)
end





# inputs format
inputs_train = hcat(inputs_train...)
println("Matriz",DataFrame(inputs_train, :auto))
println("Tamaño matriz", size(inputs_train))

# targets operations
targets_train = [row[TARGETS] for row in eachrow(data)]






# Define los hiperparámetros del modelo (ajústalos según tus necesidades)
modelHyperparameters = Dict(
    "n_neighbors" => 2,  # Ejemplo de topología de red neuronal
)

# Define el número de folds para la validación cruzada
num_folds = 5

# # Genera los índices de validación cruzada
# crossValidationIndices = crossvalidation(targets_train, num_folds)

# # Guarda los índices en un archivo
# writedlm("crossValidationIndices.csv", crossValidationIndices, ',')

# Cargar los índices desde el archivo
crossValidationIndices = readdlm("crossValidationIndices.csv", ',', Int64)[:]

println("crossValidationIndices: ", crossValidationIndices)
println("Nº de índices: ", size(crossValidationIndices)) # Debería ser igual al número de instancias


# Aplica modelCrossValidation en la GPU
resultados = gpu(modelCrossValidation)(:KNeighborsClassifier, modelHyperparameters, inputs_train, targets_train, crossValidationIndices)
println(resultados)