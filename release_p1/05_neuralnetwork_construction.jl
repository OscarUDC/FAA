#2
using Flux


"""
Creates a neuronal network with the depth chosen, and with the number of neurons chosen
----------
Attributes
----------
numInputs: number of inputs the neuronal network will recieve.
topology: number of neurones each layer has. 
numOutputs: number of outputs the neuronal network has.
transferFunctions: which transfer fuction each layer has.
"""
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    
    ann = Chain()
    for index in eachindex(topology)
        if index == 1
            ann = Chain(ann..., Dense(numInputs, topology[index], transferFunctions[index]))
        else
            ann = Chain(ann..., Dense(topology[index - 1], topology[index], transferFunctions[index]))
        end
    end
    if numOutputs > 2
        ann = Chain(ann..., Dense(topology[end], numOutputs), softmax)
    else
        ann = Chain(ann..., Dense(topology[end], 1, transferFunctions[end]))
    end
    return ann
end;