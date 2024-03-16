include("03_classification.jl")

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        return accuracy(outputs[:], targets[:]) 
    else
        classComparison = outputs .== targets
        correctClassifications = all(classComparison, dims = 2)
        mean(correctClassifications)
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return accuracy(outputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    outputs = classifyOutputs(outputs; threshold)
    return accuracy(outputs, targets)
end;