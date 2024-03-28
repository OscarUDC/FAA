#2
using Flux.Losses
using Flux

"""
Creates and trains a neuronal network with the depth and the number of neurons chosen
----------
Attributes
----------
topology: number of neurones each layer has. 
Inputs: inputs the neuronal network will recieve.
targets: the outputs the ANN is supposed to return.
transferFunctions: which transfer fuction each layer has.
maxEpochs: maximum number of epoches (times the training bucle) can have the ANN
minLoss: point where the ANN is fully "trained"
learningRate: learning rate of the training fuction
"""
function trainClassANN(topology::AbstractArray{<:Int,1},  
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),  
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01)

    newTargets = reshape(targets, :, 1)
    
    trainClassANN(topology, Tuple{inputs, newTargets}, transferFunctions, maxEpochs, minLoss, learningRate)
end

"""
Creates and trains a neuronal network with the depth and the number of neurons chosen
----------
Attributes
----------
topology: number of neurones each layer has. 
Inputs: inputs the neuronal network will recieve.
targets: the outputs the ANN is supposed to return.
transferFunctions: which transfer fuction each layer has.
maxEpochs: maximum number of epoches (times the training bucle) can have the ANN
minLoss: point where the ANN is fully "trained"
learningRate: learning rate of the training fuction
"""
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

    ann = buildClassANN(numInputs=size(inputs, 1), topology, numOutputs=size(targets, 1), transferFunctions)
    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets) : Losses.crossEntropy(model(inputs), targets)

    opt = Adam(learningRate)
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt)
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann
        end
    end
    return ann
end

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
    for epoch in 1:maxEpochs  
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