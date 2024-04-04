
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Random
using Flux
using Flux.Losses


# -------------------------------------------------------------------------
# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes) == 2                 # Si el número de clases es mayor que 2, entonces:
        encoded_matrix = reshape(feature .== classes[1], :, 1) # Para el caso de 2 clases, se crea un vector columna con 1 en las filas donde la característica es igual a la primera clase y 0 en las demás.
    else
        encoded_matrix = permutedims(classes) .== feature
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
function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    means, stds = normalizationParameters
    return (dataset .- means) ./ stds

end;
function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
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
    ann = Chain()  # Inicializa la red neuronal
    # Agrega las capas intermedias
    for index in eachindex(topology)
        if index == 1
            ann = Chain(ann..., Dense(numInputs, topology[index], transferFunctions[index]))
        else
            ann = Chain(ann..., Dense(topology[index - 1], topology[index], transferFunctions[index]))
        end
    end
    # Agrega la capa de salida
    if numOutputs > 2
        ann = Chain(ann..., Dense(topology[end], numOutputs), softmax)
    else
        ann = Chain(ann..., Dense(topology[end], 1, transferFunctions[end]))
    end
    return ann
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    
    inputs, targets = dataset
    inputs = Float32.(inputs)
    targets = Float32.(targets)
    ann = buildClassANN(Int64(size(inputs, 1)), topology, Int64(size(targets, 1));
    transferFunctions = transferFunctions)
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    opt = Adam(learningRate)
    for _ in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs', targets')], opt)
        if loss(ann, inputs', targets') <= minLoss
            return ann
        end
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),  maxEpochs::Int=1000,
    minLoss::Real=0.0, learningRate::Real=0.01)

    columnTargets = reshape(targets, :, 1)
    trainClassANN(topology, Tuple{inputs, columnTargets}, transferFunctions, maxEpochs, minLoss, learningRate)
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
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    maxEpochsVal::Int=20)
    
    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset
    validationInputs, validationTargets = validationDataset
    trainingInputs = Float32.(trainingInputs)
    trainingTargets = Float32.(trainingTargets)
    testInputs = Float32.(testInputs)
    testTargets = Float32.(testTargets)
    validationInputs = Float32.(validationInputs)
    validationTargets = Float32.(validationTargets)
    existsTestDataset = testDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
    existsValidationDataset = validationDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))

    ann = buildClassANN(Int64(size(trainingInputs, 1)),topology, Int64(size(trainingTargets, 1));
    transferFunctions = transferFunctions)
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    opt = Adam(learningRate)

    trainingLossHistory = Float32[]
    testLossHistory = Float32[]
    validationLossHistory = Float32[]

    bestANN = deepcopy(ann)
    bestANNError = Inf
    epochs = 0
    epoch = 0
    trainingLoss = minLoss + 1
    while !(epoch == maxEpochs) || !(epochs == maxEpochsVal) || !(trainingLoss <= minLoss)
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt)
        trainingLoss = loss(ann, trainingInputs', trainingTargets')
        push!(trainingLossHistory, loss(ann, trainingInputs', trainingTargets'))
        if existsTestDataset
            push!(testLossHistory, loss(ann, testInputs', testTargets'))
        end
        if existsValidationDataset
            validationLoss = loss(ann, validationInputs', validationTargets')
            push!(validationLossHistory, validationLoss)
            if validationLoss < bestANNError
                bestANN = deepcopy(ann)
                bestANNError = validationLoss
                epochs = 0
            else 
                epochs += 1
            end
        end
        epoch += 1
    end
    if existsTestDataset & existsValidationDataset
        return ann, trainingLossHistory
    elseif existsTestDataset
        return ann, trainingLossHistory, testLossHistory
    elseif  existsValidationDataset
        return bestANN, trainingLossHistory, validationLossHistory
    else 
        return bestANN, trainingLossHistory, validationLossHistory, testLossHistory
    end       
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset
    validationInputs, validationTargets = validationDataset
    trainingTargets = reshape(trainingTargets, :, 1)
    testTargets = reshape(testTargets, :, 1)
    validationTargets = reshape(validationTargets, :, 1)

    # Llamamos a la función original con los nuevos argumentos
    return trainClassANN(topology, (trainingInputs, trainingTargets);
    validationDataset =(validationInputs, validationTargets),
    testDataset=(testInputs, testTargets), transferFunctions=transferFunctions, maxEpochs=maxEpochs,
    minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    accuracy = (VP + VN + FP + FN) == 0 ? 0.0 : (VN + VP) / (VP + VN + FP + FN)
    error_rate = (VP + VN + FP + FN) == 0 ? 0.0 : (FN + FP) / (VP + VN + FP + FN)
    sensitivity = (VP + FN) == 0 ? 1.0 : VP / (FN + VP)
    specificity = (VN + FP) == 0 ? 1.0 : VN / (FP + VN)
    ppv = (VP + FP) == 0 ? 1.0 : VP / (VP + FP)
    npv = (VN + FN) == 0 ? 1.0 : VN / (VN + FN)
    f1_score = (sensitivity + ppv) == 0 ? 0.0 : 2 * (ppv * sensitivity) / (ppv + sensitivity)
    confusion_matrix = [VN FP; FN VP]
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
    _, n_classes = size(outputs)  # Obtener las dimensiones de las matrices
    if n_classes != size(targets, 2)  # Verificar si el número de columnas es diferente en las matrices outputs y targets
        error("Number of columns in outputs and targets matrices must be equal")
    end

    if n_classes == 1  # Si solo hay una clase, llamar a la función anterior
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end
    # Inicializar vectores de métricas con ceros
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)
    # Calcular matriz de confusión(sin necesidad de inicializarla primero con ceros)
    for class in axes(outputs, 2)
        _, _, sensitivity[class], specificity[class], VPP[class], VPN[class], F1[class], _ =
        confusionMatrix(outputs[:, class], targets[:, class])
    end
    acc = accuracy(outputs, targets)
    confusion_matrix = zeros(n_classes, n_classes)
    for row in axes(outputs, 1)
        realClass = findfirst(targets[row, :])
        predictedClass = findfirst(outputs[row, :])
        confusion_matrix[realClass, predictedClass] += 1
    end
    if !weighted
        return(acc, 1 - acc, mean(sensitivity), mean(specificity), mean(VPP), mean(VPN), mean(F1), confusion_matrix)
    else
        ponderation = sum(confusion_matrix, dims = 2) ./ size(targets, 1)
        sensitivity .*= ponderation
        specificity .*= ponderation
        VPP .*= ponderation
        VPN .*= ponderation
        F1 .*= ponderation
    return (acc, 1 - acc, sum(sensitivity), sum(specificity), sum(VPP), sum(VPN), sum(F1), confusion_matrix)
    end
end;

# Definir función confusionMatrix para matrices de valores reales
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Convertir outputs a valores booleanos si es necesario
    outputs_bool = classifyOutputs(outputs)
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end;

# Definir función confusionMatrix para vectores de cualquier tipo
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurarse de que todas las clases del vector de outputs estén incluidas en el vector de targets
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
    indexes = zeros(Int64, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indexes = zeros(Int, size(targets, 1))
    for class in axes(targets, 2)
        indexes[targets[:, class] .== true] = crossvalidation(sum(targets[:, class]), k)
    end
    return indexes
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    if length(unique(targets)) > 2
        processedTargets = oneHotEncoding(targets)
    else
        processedTargets = targets .== unique(targets)[1]
    end
    indexes = crossvalidation(processedTargets, k)
    return indexes
end;







function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    
    numFolds = maximum(crossValidationIndices)
    processedTargets = oneHotEncoding(targets)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = processedTargets[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = processedTargets[crossValidationIndices .== fold, :]
        (accuracyExecution, errorRateExecution, sensitivityExecution, specificityExecution,
        ppvExecution, npvExecution, f1ScoreExecution) = (Vector{Float64}(undef, numExecutions) for _ in 1:7)
        for execution in 1:numExecutions
            if validationRatio > 0
                N = size(trainingInputs, 1)
                trainingIndexes, validationIndexes = holdOut(N, validationRatio)
                newTrainingInputs = trainingInputs[trainingIndexes, :]
                validationInputs = trainingInputs[validationIndexes, :]
                newTrainingTargets = trainingTargets[trainingIndexes, :]
                validationTargets = trainingTargets[validationIndexes, :]
                ann, _... = trainClassANN(topology, (newTrainingInputs, newTrainingTargets);
                validationDataset = (validationInputs, validationTargets), testDataset = (testInputs, testTargets),
                transferFunctions = transferFunctions, maxEpochs = maxEpochs, maxEpochsVal = maxEpochsVal, minLoss = minLoss,
                learningRate = learningRate)
            else
                ann, _... = trainClassANN(topology, (trainingInputs, trainingTargets);
                transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss,
                learningRate = learningRate)
            end
            testOutputs = ann(testInputs')
            accuracyExecution[execution], errorRateExecution[execution], sensitivityExecution[execution],
            specificityExecution[execution], ppvExecution[execution], npvExecution[execution],
            f1ScoreExecution[execution] = confusionMatrix(testOutputs', testTargets)
        end
        accuracy[fold] = mean(accuracyExecution)
        errorRate[fold] = mean(errorRateExecution)
        sensitivity[fold] = mean(sensitivityExecution)
        specificity[fold] = mean(specificityExecution)
        ppv[fold] = mean(ppvExecution)
        npv[fold] = mean(npvExecution)
        f1Score[fold] = mean(f1ScoreExecution)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)),
    (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
    (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1Score), std(f1Score)))
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 6 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    if modelType == :ANN
        if !haskey(modelHyperparameters, "numExecutions")
            modelHyperparameters["numExecutions"] = 50
        end
        if !haskey(modelHyperparameters, "transferFunctions")
            modelHyperparameters["transferFunctions"] = fill(σ, length(modelHyperparameters["topology"]))
        end
        if !haskey(modelHyperparameters, "maxEpochs")
            modelHyperparameters["maxEpochs"] = 1000
        end
        if !haskey(modelHyperparameters, "minLoss")
            modelHyperparameters["minLoss"] = 0.0
        end
        if !haskey(modelHyperparameters, "learningRate")
            modelHyperparameters["learningRate"] = 0.01
        end
        if !haskey(modelHyperparameters, "validationRatio")
            modelHyperparameters["validationRatio"] = 0
        end
        if !haskey(modelHyperparameters, "maxEpochsVal")
            modelHyperparameters["maxEpochsVal"] = 20
        end
        return ANNCrossValidation(modelHyperparameters["topology"], inputs, targets, crossValidationIndices;
        numExecutions = modelHyperparameters["numExecutions"],
        transferFunctions = modelHyperparameters["transferFunctions"],
        maxEpochs = modelHyperparameters["maxEpochs"], minLoss = modelHyperparameters["minLoss"],
        learningRate = modelHyperparameters["learningRate"], validationRatio = modelHyperparameters["validationRatio"],
        maxEpochsVal = modelHyperparameters["maxEpochsVal"])
    end 
    processedTargets = string.(targets)
    numFolds = maximum(crossValidationIndices)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = processedTargets[crossValidationIndices .!= fold]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = processedTargets[crossValidationIndices .== fold]
        if modelType == :SVC
            if !haskey(modelHyperparameters, "kernel")
                modelHyperparameters["kernel"] = "rbf"
            end
            if !haskey(modelHyperparameters, "degree")
                modelHyperparameters["degree"] = 3
            end
            if !haskey(modelHyperparameters, "gamma")
                modelHyperparameters["gamma"] = "scale"
            end
            if !haskey(modelHyperparameters, "coef0")
                modelHyperparameters["coef0"] = 0.0
            end
            model = SVC(C = modelHyperparameters["C"], kernel = modelHyperparameters["kernel"],
            degree = modelHyperparameters["degree"], gamma = modelHyperparameters["gamma"],
            coef0 = modelHyperparameters["coef0"])
        elseif modelType == :DecisionTreeClassifier
            if !haskey(modelHyperparameters, "max_depth")
                modelHyperparameters["max_depth"] = nothing
            end
            model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"], random_state = 1)
        elseif modelType == :KNeighborsClassifier
            if !haskey(modelHyperparameters, "n_neighbors")
                modelHyperparameters["n_neighbors"] = 5
            end
            model = KNeighborsClassifier(n_neighbors = modelHyperparameters["n_neighbors"])
        else
            throw(ArgumentError("Model type $modelType does not exist or is not allowed"))
        end
        trainedModel = fit!(model, trainingInputs, trainingTargets)
        testOutputs = predict(trainedModel, testInputs)
        accuracy[fold], errorRate[fold], sensitivity[fold], specificity[fold],
        ppv[fold], npv[fold], f1Score[fold] = confusionMatrix(testOutputs, testTargets)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)), (mean(sensitivity),
    std(sensitivity)), (mean(specificity), std(specificity)), (mean(ppv), std(ppv)), (mean(npv), std(npv)),
    (mean(f1Score), std(f1Score)))
end;
