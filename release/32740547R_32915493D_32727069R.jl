
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    num_patterns = length(feature)

    if num_classes == 2
        encoded_matrix = reshape(feature .== classes[1], :, 1)
    else
        encoded_matrix = falses(num_patterns, num_classes)

        for i in 1:num_classes
            encoded_matrix[:, i] .= feature .== classes[i]
        end
    end

    return encoded_matrix
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)                                              # Obtenemos las clases únicas presentes en el vector de características
    return oneHotEncoding(feature, classes)                                # Llamamos al método con la información de clases obtenida

end;
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return hcat(feature)                                                   # Convertimos el vector booleano en una matriz columna

end;
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente


# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins, maxs)

end;
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    means = mean(dataset, dims=1)
    stds = std(dataset, dims=1)
    return (means, stds)

end;

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mins, maxs = normalizationParameters
    dataset .= (dataset .- mins) ./ (maxs .- mins)

end;
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    dataset .= (dataset .- mins) ./ (maxs .- mins)

end;
function normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mins, maxs = normalizationParameters
    return (dataset .- mins) ./ (maxs .- mins)

end;
function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (dataset .- mins) ./ (maxs .- mins)

end;

# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    means, stds = normalizationParameters
    dataset .-= means
    dataset ./= stds

end;
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    means = mean(dataset, dims=1)
    stds = std(dataset, dims=1)
    dataset .-= means
    dataset ./= stds

end;
function normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    means, stds = normalizationParameters
    return (dataset .- means) ./ stds

end;
function normalizeZeroMean( dataset::AbstractArray{<:Real,2})
    means = mean(dataset, dims=1)
    stds = std(dataset, dims=1)
    return (dataset .- means) ./ stds

end;


# -------------------------------------------------------
# Funcion que permite transformar una matriz de valores reales con las salidas del clasificador o clasificadores en una matriz de valores booleanos con la clase en la que sera clasificada

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        rowwise_outputs = classifyOutputs(outputs[:]; threshold)
        return reshape(rowwise_outputs, :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end
end;


# -------------------------------------------------------
# Funciones para calcular la precision

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end;
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        return accuracy(outputs[:], targets[:]) 
    else
        classComparison = outputs .== targets
        correctClassifications = all(classComparison, dims = 2)
        mean(correctClassifications)
    end
end;
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return accuracy(outputs, targets)
end;
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    outputs = classifyOutputs(outputs; threshold)
    return accuracy(outputs, targets)
end;

# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    ann = Chain()                                                                                           #initializes the ANN
    ann = Chain(ann..., Dense(numInputs, topology[1],transferFunctions[1]))                                 #first layer of the ANN
    number_before = 1
   
    for number in 2:1: length(topology)                                                                    #loop that will create
        ann = Chain(ann..., Dense(topology[number_before], topology[number],transferFunctions[number]))     #the other layers of the ANN
        number_before = number
    end
    if numOutputs > 2
        ann = Chain(ann..., Dense(topology[number_before], numOutputs, identity))                               #lasts layers of the ANN
        ann = Chain(ann..., softmax)
    else
        ann = Chain(ann..., Dense(topology[number_befure], 1, transferFunctions[number_before]))
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    
    inputs = dataset[1]
    targets = dataset[2]
    inputsT = transpose(inputs)
    targetsT = transpose(targets)
    
    ann = buildClassANN(size(inputs, 1), topology, size(targets, 1), transferFunctions)
    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets) : Losses.crossEntropy(model(inputs), targets)

    opt_state = Flux.setup(Adam(learningRate), ann)
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt_state)
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann
        end
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    newTargets = reshape(targets, 1, size(targets))
    
    trainClassANN(topology, Tuple{inputs, newTargets}, transferFunctions, maxEpochs, minLoss, learningRate)
end

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    if P < 0 || P > 1
        error("P must be in the interval [0, 1]")
    end
    indexes = randperm(N)
    inTest = Int(round(P * N))
    inTrain = N - inTest
    return (indexes[1:inTrain], indexes[inTrain + 1:end])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    if (Pval < 0 || Pval > 1) || (Ptest < 0 || Ptest > 1) || (Pval + Ptest > 1)
        error("Pval and Ptest must be in the interval [0, 1], and Pval + Ptest can't be greater than 1")
    end
    trainIndexes, otherIndexes = holdOut(N, Pval + Ptest)
    newN = length(otherIndexes)
    inTest = Int(floor(newN * Ptest / (Pval + Ptest)))
    inVal = newN - inTest
    return (trainIndexes, otherIndexes[1:inVal], otherIndexes[inVal + 1:end])
end;


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test. Estos dos ultimos son opcionales
# Es la funcion anterior, modificada para calcular errores en los conjuntos de validacion y test y realizar parada temprana si es necesario
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)            # Verdaderos positivos
    VN = sum((.!outputs) .& (.!targets))    # Verdaderos negativos
    FP = sum(outputs .& .!targets)  # Falsos positivos
    FN = sum(.!outputs .& targets)  # Falsos negativos

    if (VP + VN + FP + FN) == 0
        accuracy = 0.0
    else
        accuracy = (VN + VP) / (VN + VP + FN + FP)
    end
    
    if (VN + VP + FN + FP) == 0
        error_rate = 0.0
    else
        error_rate = (FN + FP) / (VN + VP + FN + FP)
    end

#-----------------------
    
    if VP == 0 && FN == 0
        sensitivity = 1.0
    else
        sensitivity = VP / (FN + VP)
    end
    
    if VN == 0 && FP == 0
        specificity = 1.0
    else
        specificity = VN / (FP + VN)
    end
    
    if VP == 0 && FP == 0
        ppv = 1.0
    else
        ppv = VP / (VP + FP)
    end

    if VN == 0 && FN == 0
        npv = 1.0
    else
        npv = VN / (VN + FN)
    end
    
    if sensitivity == 0 && prec == 0
        f1_score = 0.0
    else
        f1_score = 2 * (accuracy * sensitivity) / (accuracy + sensitivity)
    end

    confusion_matrix = [VP FN; FP VN]

    return (accuracy, error_rate, sensitivity, specificity, ppv, npv, f1_score, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    return confusionMatrix(binary_outputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    results = confusionMatrix(outputs, targets)
    
    println("accuracy: ", results[1])
    println("Error Rate: ", results[2])
    println("Sensitivity: ", results[3])
    println("Specificity: ", results[4])
    println("PPV: ", results[5])
    println("NPV: ", results[6])
    println("F1-Score: ", results[7])
    println("Confusion Matrix:")
    println(results[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    printConfusionMatrix(binary_outputs, targets)
end;



# Clasificación multiclase:


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    n_classes = size(outputs, 2)
    n_patterns = size(outputs, 1)
    if n_classes <= 2 || n_patterns <= 2
        error("Invalid input dimensions")
    end
    
    # Verificar si solo hay una columna y llamar a la función anterior
    if n_classes == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end

    # Inicialización de vectores para métricas
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)

    # Cálculo de métricas para cada clase
    for i in 1:n_classes
        outputs_class = outputs[:, i]
        targets_class = targets[:, i]
        class_metrics = confusionMatrix(outputs_class, targets_class)
        sensitivity[i] = class_metrics[3]
        specificity[i] = class_metrics[4]
        VPP[i] = class_metrics[5]
        VPN[i] = class_metrics[6]
        F1[i] = class_metrics[7]
    end

    # Cálculo de métricas macro o weighted
    if weighted
        weights = sum(targets, dims=1)
        prec = dot(VPP, weights) / sum(weights)
        error_rate = 1 - prec
        sensitivity = dot(sensitivity, weights) / sum(weights)
        specificity = dot(specificity, weights) / sum(weights)
        VPP = dot(VPP, weights) / sum(weights)
        VPN = dot(VPN, weights) / sum(weights)
        F1 = dot(F1, weights) / sum(weights)
    else
        prec = accuracy(outputs, targets)
        error_rate = 1 - prec
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        VPP = mean(VPP)
        VPN = mean(VPN)
        F1 = mean(F1)
    end

    # Cálculo de matriz de confusión
    confusion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:n_classes, j in 1:n_classes]

    return (prec, error_rate, sensitivity, specificity, VPP, VPN, F1, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todos los elementos de outputs estén presentes en targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end;

# Función para imprimir la matriz de confusión
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;

# Función para imprimir la matriz de confusión con entradas de tipo AbstractArray{<:Real,2}
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;

# Función para imprimir la matriz de confusión con entradas de tipo AbstractArray{<:Any,1}
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    if k > N
        error("k cannot be greater than N")
    end
    subset = collect(1:k)
    subsets = repeat(subset, outer = ceil(Int, N/k))
    return shuffle!(subsets[1:N])
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indexes = zeros(Int, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;







function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end;
