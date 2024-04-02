#5 and 6
using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    if k > N
        error("k cannot be greater than N")
    end
    subset = collect(1:k)
    subsets = repeat(subset, outer = ceil(Int, N/k))
    return shuffle!(subsets[1:N])
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    if k < 10
        print("WARNING: K is very low, may affect results")
    indexes = zeros(Int, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    if k < 10
        print("WARNING: K is very low, may affect results")
    indexes = zeros(Int, size(targets, 1))
    for class in size(targets, 2)
        indexes[class] = crossvalidation(sum(targets[:, class]), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    if size(targets, 2) > 2
        targets = oneHotEncoding(targets)
    indexes = zeros(Int, size(targets, 1))
    for class in size(targets, 2)
        indexes[class] = crossvalidation(sum(targets[:, class]), k)
    return indexes
end;