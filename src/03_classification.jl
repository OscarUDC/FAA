function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        rowwise_output = classifyOutputs(outputs[:]; threshold)
        return reshape(rowwise_output, :, 1)
    end
end;