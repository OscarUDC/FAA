using Flux


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann = Chain();
    ann = Chain(ann..., Dense(numInputs, topology[0],transferFunctions[0]));
    number_before = 0;
    for number in 1:1:length(topology)
        ann = Chain(ann..., Dense(topology[number_before], topology[number],transferFunctions[number]));
        number_before = number;
    end
    ann = Chain(ann..., Dense(topology[number_before], numOutputs, transferFunctions[number_before]));
    ann = Chain(ann..., softmax);
    return ann;
end

