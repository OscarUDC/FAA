function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        rowwise_outputs = classifyOutputs(outputs[:]; threshold)
        return reshape(rowwise_outputs, :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end
end;