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
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int,  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
   
    ann = Chain()                                                                                           #initializes the ANN
    ann = Chain(ann..., Dense(numInputs, topology[1],transferFunctions[1]))                                 #first layer of the ANN
    number_before = 1
   
    for number in 2: length(topology)                                                                    #loop that will create
        ann = Chain(ann..., Dense(topology[number_before], topology[number],transferFunctions[number]))     #the other layers of the ANN
        number_before = number
    end
    ann = Chain(ann..., Dense(topology[number_before], numOutputs, identity))                               #last layer of the ANN
    ann = Chain(ann..., softmax)                                                                            #the softmax function
    return ann
end