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
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    
    ann = Chain()  # Inicializa la red neuronal

    # Agrega las capas intermedias
    for i in 1:1:length(topology)
        if i == 1
            ann = Chain(ann..., Dense(numInputs, topology[i], transferFunctions[i]))
        else
            ann = Chain(ann..., Dense(topology[i-1], topology[i], transferFunctions[i]))
        end
    end

    # Agrega la capa de salida
    if numOutputs > 2
        ann = Chain(ann..., Dense(topology[end], numOutputs, identity), softmax)
    else
        ann = Chain(ann..., Dense(topology[end], 1, transferFunctions[end]))
    end
    
    return ann
end