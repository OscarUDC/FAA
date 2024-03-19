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

    newTargets = reshape(targets, :, size(targets))
    
    trainClassANN(topology, Tuple{inputs, newTargets}, transferFunctions, maxEpochs, minLoss, learningRate)
end

"""
Creation of the loss function, please the matrix has to have arranged the patrons in colums
----------
Attributes
----------
model: model to calculate the loss
inputs: the inputs you'll use to calculate the loss
targets: the targets youll use to calculate the loss        
"""
loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets) : Losses.crossEntropy(model(inputs), targetsT)


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

    inputs = dataset[0]
    targets = dataset[1]
    inputsT = transpose(inputs)
    targetsT = transpose(targets)
    ann = buildClassANN(numInputs=size(inputs, 1), topology, numOutputs=size(targets, 1), transferFunctions)

    opt_state = Flux.setup(Adam(learningRate), ann)
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt_state)
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann
        end
    end
    return ann
end