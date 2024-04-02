
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
    
    ann = Chain()  # Inicializa la red neuronal

    # Agrega las capas intermedias
    for i in 1:1:length(topology)
        if i == 1
            ann = Chain(ann, Dense(numInputs, topology[i], transferFunctions[i]))
        else
            ann = Chain(ann, Dense(topology[i-1], topology[i], transferFunctions[i]))
        end
    end

    # Agrega la capa de salida
    if numOutputs > 2
        ann = Chain(ann, Dense(topology[end], numOutputs, identity), softmax)
    else
        ann = Chain(ann, Dense(topology[end], 1, transferFunctions[end]))
    end
    
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000,
    minLoss::Real=0.0,
    learningRate::Real=0.01)
    
    inputs, targets = dataset

    inputsT = transpose(inputs)
    targetsT = transpose(targets)
    
    inputsT = Float32.(inputsT)
    targetsT = Float32.(targetsT)

    ann = buildClassANN(Int64(size(inputs, 1)), topology, Int64(size(targets, 1)))
    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets) : Losses.crossEntropy(model(inputs), targets)

    opt = Adam(learningRate)
    for epoch in 0:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt)
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann
        end
    end
    return ann
end

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

newTargets = reshape(targets, 1, length(targets))

return trainClassANN(topology, Tuple{inputs, newTargets}, transferFunctions, maxEpochs, minLoss, learningRate)
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
    
   
    # Extraemos los datos de entrada y salida de los conjuntos de datos
    inputsTr, targetsTr = trainingDataset
    inputsV, targetsV = validationDataset
    inputsTe, targetsTe = testDataset
    
    # Transponemos las matrices de entrada y salida para cada dataset si es necesario
    inputsTr = transpose(inputsTr)
    targetsTr = transpose(targetsTr)

    inputsV = transpose(inputsV)
    targetsV = transpose(targetsV)

    inputsTe = transpose(inputsTe)
    targetsTe = transpose(targetsTe)

    # Convertimos los datos a Float32
    inputsTr = Float32.(inputsTr)
    targetsTr = Float32.(targetsTr)

    inputsV = Float32.(inputsV)
    targetsV = Float32.(targetsV)

    inputsTe = Float32.(inputsTe)
    targetsTe = Float32.(targetsTe)

    # Construimos la red neuronal
    ann = buildClassANN(size(inputsTr, 1),topology, size(targetsTr, 1))

    # Definimos la función de pérdida
    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets) : Losses.crossEntropy(model(inputs), targets)

    # Inicializamos el optimizador
    opt = ADAM(learningRate)

    # Vector para almacenar los valores de pérdida en cada época
    lossHistoryTr = Float32[]
    lossHistoryV = Float32[]
    lossHistoryTe = Float32[]

    mejorRed = deepcopy(ann)
    errorMejorRed = Inf
    ciclos = 0
    # Bucle de entrenamiento
    for epoch in 0:maxEpochs  
        # Calculamos el valor de pérdida en cada época
        currentLossTr = loss(ann, inputsTr, targetsTr)
        push!(lossHistoryTr, currentLossTr)
        
        if testDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
            currentLossTe = loss(ann, inputsTe, targetsTe)
            push!(lossHistoryTe, currentLossTe)
        end

        Flux.train!(loss, Flux.params(ann), [(inputsTr, targetsTr)], opt)

        if validationDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
            currentLossV = loss(ann, inputsV, targetsV)
            push!(lossHistoryV, currentLossV)
            if currentLossV < errorMejorRed
                mejorRed = deepcopy(ann)
                errorMejorRed = currentLossV
                ciclos = 0
            else 
                ciclos += 1
                if ciclos == maxEpochsVal
                    break
                end 
            end
        end
        # Verificamos los criterios de parada
        if currentLossTr ≤ minLoss
            println("Entrenamiento completado en la época ", epoch)
            break
        end
    end

    # devolver unos valores u otros en funcion de los parametros pasados
    if testDataset == validationDataset == (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
        return ann, lossHistoryTr
    elseif testDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)) && validationDataset == (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
        return ann, lossHistoryTr, lossHistoryTe
    elseif  validationDataset != (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)) && testDataset == (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0))
        return mejorRed, lossHistoryTr, lossHistoryV
    else 
        return mejorRed, lossHistoryTr, lossHistoryV, lossHistoryTe
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

    inputs_train, targets_train = trainingDataset
    inputs_val, targets_val = validationDataset
    inputs_test, targets_test = testDataset
    
    # Convertir las salidas deseadas de matrices a vectores
    targets_train = reshape(trainingDataset[2], :, 1)
    targets_val = reshape(validationDataset[2], :, 1)
    targets_test = reshape(testDataset[2], :, 1)

    # Llamamos a la función original con los nuevos argumentos
    return trainClassANN(topology, (inputs_train, targets_train); validationDataset =(inputs_val, targets_val), testDataset=(inputs_test, targets_test), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------



function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # Precisión
    if (VP + VN + FP + FN) == 0
        accuracy = 0.0
    else
        accuracy = (VN + VP) / (VN + VP + FN + FP)
    end
    
    # Tasa de error
    if (VP + VN + FP + FN) == 0
        error_rate = 0.0
    else
        error_rate = (FN + FP) / (VN + VP + FN + FP)
    end
    
    # Sensibilidad (Recall)
    if VP == 0 && FN == 0
        sensitivity = 1.0
    else
        sensitivity = VP / (FN + VP)
    end
    
    # Especificidad
    if VN == 0 && FP == 0
        specificity = 1.0
    else
        specificity = VN / (FP + VN)
    end
    
    # Valor predictivo positivo (Precision)
    if VP == 0 && FP == 0
        ppv = 1.0
    else
        ppv = VP / (VP + FP)
    end

    # Valor predictivo negativo
    if VN == 0 && FN == 0
        npv = 1.0
    else
        npv = VN / (VN + FN)
    end
    
    # F1-score
    if sensitivity == 0 && ppv == 0
        f1_score = 0.0
    else
        f1_score = 2 * (ppv * sensitivity) / (ppv + sensitivity)
    end    

    # Matriz de confusión
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
    n_patterns, n_classes = size(outputs)  # Obtener las dimensiones de las matrices
    if n_classes != size(targets, 2)  # Verificar si el número de columnas es diferente en las matrices outputs y targets
        error("Number of columns in outputs and targets matrices must be equal")
    end

    if n_classes == 1  # Si solo hay una clase, llamar a la función anterior
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end

    if n_classes == 2  # Comprobar si el número de clases es igual a 2
        error("Invalid input dimensions")
    end

    # Inicializar vectores de métricas con ceros
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)

    # Calcular matriz de confusión(sin necesidad de inicializarla primero con ceros)
    confusion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:n_classes, j in 1:n_classes] #Hacemos un doble bucle para rellenar la matriz con ceros de primeras, con el fin de reservar los huecos en memoria

    # Calcular métricas macro o weighted
    if weighted
        weights = sum(targets, dims=1)
        TP = sum(outputs .& targets, dims=1)
        TN = sum((.!outputs) .& (.!targets), dims=1)
        FP = sum(outputs .& .!targets, dims=1)
        FN = sum(.!outputs .& targets, dims=1)

        accuracy = sum(TP ./ (TP .+ FP) .* weights) / sum(weights)
        error_rate = 1 - accuracy
        sensitivity = sum(TP ./ (TP .+ FN) .* weights) / sum(weights)
        specificity = sum(TN ./ (TN .+ FP) .* weights) / sum(weights)
        VPP = sensitivity  # Same as sensitivity for multiclass
        VPN = specificity  # Same as specificity for multiclass
        F1 = 2 * sensitivity * accuracy / (sensitivity + accuracy)

    else
        TP = sum(outputs .& targets, dims=1)
        TN = sum((.!outputs) .& (.!targets), dims=1)
        FP = sum(outputs .& .!targets, dims=1)
        FN = sum(.!outputs .& targets, dims=1)

        accuracy = sum(TP ./ (TP .+ FP)) / n_classes
        error_rate = 1 - accuracy
        sensitivity = sum(TP ./ (TP .+ FN)) / n_classes
        specificity = sum(TN ./ (TN .+ FP)) / n_classes
        VPP = sensitivity  # Same as sensitivity for multiclass
        VPN = specificity  # Same as specificity for multiclass
        F1 = 2 * sensitivity * accuracy / (sensitivity + accuracy) / n_classes
    end

    # Unir los valores de métricas para cada clase en un único valor usando la estrategia macro o weighted
    sensitivity = weighted ? sensitivity : mean(sensitivity)
    specificity = weighted ? specificity : mean(specificity)
    VPP = weighted ? VPP : mean(VPP)
    VPN = weighted ? VPN : mean(VPN)
    F1 = weighted ? F1 : mean(F1)

    return (accuracy, error_rate, sensitivity, specificity, VPP, VPN, F1, confusion_matrix)
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
    indexes = zeros(Int, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indexes = zeros(Int, size(targets, 1))
    for class in axes(targets, 2)
        indexes[class] = crossvalidation(sum(targets[:, class]), k)
    end
    return indexes
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    if size(targets, 2) > 2
        targets = oneHotEncoding(targets)
    indexes = zeros(Int, size(targets, 1))
    for class in axes(targets, 2)
        indexes[class] = crossvalidation(sum(targets[:, class]), k)
    end
    return indexes
end;







function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    
    numFolds = maximum(crossValidationIndices)
    inputs = oneHotEncoding(inputs)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = targets[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = targets[crossValidationIndices .== fold, :]
        (accuracyExecution, errorRateExecution, sensitivityExecution, specificityExecution, ppvExecution, npvExecution, f1ScoreExecution) = (Vector{Float64}(undef, numExecutions) for _ in 1:7)
        for execution in 1:numExecutions
            if validationRatio > 0
                N = size(trainingInputs, 1)
                trainingIndexes, validationIndexes = holdOut(N, validationRatio)
                newTrainingInputs = trainingInputs[trainingIndexes, :]
                validationInputs = trainingInputs[validationIndexes, :]
                newTrainingTargets = trainingTargets[trainingIndexes, :]
                validationTargets = trainingTargets[validationIndexes, :]
                ann, other... = trainClassANN(topology, (newTrainingInputs, newTrainingTargets); validationDataset = (validationInputs, validationTargets), testDataset = (testInputs, testTargets), transferFunctions = transferFunctions, maxEpochs = maxEpochs, maxEpochsVal = maxEpochsVal, minLoss = minLoss, learningRate = learningRate)
            else
                ann, other... = trainClassANN(topology, (newTrainingInputs, newTrainingTargets); transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate)
            end
            testOutputs = ann(testInputs')
            accuracyExecution[execution], errorRateExecution[execution], sensitivityExecution[execution], specificityExecution[execution], ppvExecution[execution], npvExecution[execution], f1ScoreExecution[execution] = confusionMatrix(testOutputs', testTargets)
        end
        accuracy[fold] = mean(accuracyExecution)
        errorRate[fold] = mean(errorRateExecution)
        sensitivity[fold] = mean(sensitivityExecution)
        specificity[fold] = mean(specificityExecution)
        ppv[fold] = mean(ppvExecution)
        npv[fold] = mean(npvExecution)
        f1Score[fold] = mean(f1ScoreExecution)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)), (mean(sensitivity), std(sensitivity)),
    (mean(specificity), std(specificity)), (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1Score), std(f1Score)))
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
