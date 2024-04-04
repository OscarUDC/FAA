#2
using Flux.Losses
using Flux

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
    opt = Flux.setup(Adam(learningRate), ann) 
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

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20) 
    
    trainingInputs, trainingTargets = trainingDataset
    testInputs, testTargets = testDataset
    validationInputs, validationTargets = validationDataset
    trainingInputs = Float32.(trainingInputs)
    trainingTargets = Float32.(trainingTargets)
    testInputs = Float32.(testInputs)
    testTargets = Float32.(testTargets)
    validationInputs = Float32.(validationInputs)
    validationTargets = Float32.(validationTargets)

    ann = buildClassANN(Int64(size(trainingInputs, 1)),topology, Int64(size(trainingTargets, 1)))

    # Definimos la función de pérdida
    loss(model, inputs, targets) = size(targets,1)==1 ? Flux.Losses.binarycrossentropy(model(inputs), targets) : Flux.Losses.crossentropy(model(inputs), targets)

    # Inicializamos el optimizador
    opt = Flux.setup(Adam(learningRate), ann)

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
    targets_train = reshape(targets_train, :, 1)
    targets_val = reshape(targets_val, :, 1)
    targets_test = reshape(targets_test, :, 1)

    # Llamamos a la función original con los nuevos argumentos
    return trainClassANN(topology, (inputs_train, targets_train); validationDataset =(inputs_val, targets_val), testDataset=(inputs_test, targets_test), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end