#2
using Flux.Losses
using Flux

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
    trainingInputs = Float32.(trainingInputs')
    trainingTargets = Float32.(trainingTargets')
    testInputs = Float32.(testInputs')
    testTargets = Float32.(testTargets')
    validationInputs = Float32.(validationInputs')
    validationTargets = Float32.(validationTargets')
    existsTestDataset = length(testDataset[2]) > 0
    existsValidationDataset = length(validationDataset[2]) > 0

    ann = buildClassANN(Int64(size(trainingInputs, 1)),topology, Int64(size(trainingTargets, 1));
    transferFunctions = transferFunctions)
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)
    opt = Flux.setup(Adam(learningRate), ann)

    trainingLossHistory = Float32[]
    testLossHistory = Float32[]
    validationLossHistory = Float32[]

    bestANN = deepcopy(ann)
    bestANNError = Inf
    epochs = 0
    epoch = 0
    trainingLoss = minLoss + 1
    while !(epoch >= maxEpochs) && !(epochs >= maxEpochsVal) && !(trainingLoss <= minLoss)
        Flux.train!(loss, ann, [(trainingInputs, trainingTargets)], opt)
        trainingLoss = loss(ann, trainingInputs, trainingTargets)
        push!(trainingLossHistory, loss(ann, trainingInputs, trainingTargets))
        if existsTestDataset
            push!(testLossHistory, loss(ann, testInputs, testTargets))
        end
        if existsValidationDataset
            validationLoss = loss(ann, validationInputs, validationTargets)
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
    return bestANN, trainingLossHistory, validationLossHistory, testLossHistory       
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