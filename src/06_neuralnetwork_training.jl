using Flux.Losses

function trainClassANN(topology::AbstractArray{<:Int,1},  
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),  
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01) 

    for epoch in 0:1:maxEpochs

    end;

end

function trainClassANN(topology::AbstractArray{<:Int,1},  
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};  
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),  
    maxEpochs::Int=1000, 
    minLoss::Real=0.0, 
    learningRate::Real=0.01) 

end