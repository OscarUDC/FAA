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
    

    inputsT = transpose(inputs);
    targetsT = transpose(targets);
    ann = buildClassANN(size(inputs, 1), topology, size(targets, 1), transferFunctions);

    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets): Losses.crossEntropy(model(inputs), targets);
    opt_state = Flux.setup(Adam(learningRate), ann);
    for epoch in 0:1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt_state);
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann;
        end;
        return ann;
    end;

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

    inputs = dataset[0]
    targets = dataset[1]
    inputsT = transpose(inputs);
    targetsT = transpose(targets);
    ann = buildClassANN(size(inputs, 1), topology, size(targets, 1), transferFunctions);

    loss(model, inputs, targets) = size(targets,1) ? Losses.binaryCrossEntropy(model(inputs), targets): Losses.crossEntropy(model(inputs), targets);
    opt_state = Flux.setup(Adam(learningRate), ann);
    for epoch in 0:1:maxEpochs
        Flux.train!(loss, ann, [(inputsT, targetsT)], opt_state);
        if loss(ann, inputsT, targetsT) <= minLoss
            return ann;
        end;
        return ann;
    end;
end;