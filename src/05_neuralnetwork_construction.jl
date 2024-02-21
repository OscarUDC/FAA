using Flux


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
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
    ann = Chain();                                                                                          #initializes the ANN
    ann = Chain(ann..., Dense(numInputs, topology[0],transferFunctions[0]));                                #first layer of the ANN
    number_before = 0;
for number in 1:1:(length(topology) - 1)                                                                         #loop that will create
        ann = Chain(ann..., Dense(topology[number_before], topology[number],transferFunctions[number]));    #the other layers of the ANN
        number_before = number;
    end
    ann = Chain(ann..., Dense(topology[number_before], numOutputs, transferFunctions[number_before]));      #last layer of the ANN
    ann = Chain(ann..., softmax);                                                                           #the softmax function
    return ann;
end