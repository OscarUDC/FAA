#5 and 6
using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    if k > N
        throw(ArgumentError("k ($k) cannot be greater than N ($N)"))
    end
    subset = collect(1:k)
    subsets = repeat(subset, outer = ceil(Int, N/k))
    return shuffle!(subsets[1:N])
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indexes = zeros(Int64, length(targets))
    indexes[targets] .= crossvalidation(sum(targets), k)
    indexes[.!targets] .= crossvalidation(sum(.!targets), k)
    return indexes
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indexes = zeros(Int, size(targets, 1))
    for class in axes(targets, 2)
        indexes[targets[:, class] .== true] = crossvalidation(sum(targets[:, class]), k)
    end
    return indexes
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    if length(unique(targets)) > 2
        processedTargets = oneHotEncoding(targets)
    else
        processedTargets = targets .== unique(targets)[1]
    end
    indexes = crossvalidation(processedTargets, k)
    return indexes
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20, showText::Bool=false)
    
    numFolds = maximum(crossValidationIndices)
    processedTargets = oneHotEncoding(targets)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = processedTargets[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = processedTargets[crossValidationIndices .== fold, :]
        (accuracyExecution, errorRateExecution, sensitivityExecution, specificityExecution,
        ppvExecution, npvExecution, f1ScoreExecution) = (Vector{Float64}(undef, numExecutions) for _ in 1:7)
        for execution in 1:numExecutions
            if validationRatio > 0
                N = size(trainingInputs, 1)
                trainingIndexes, validationIndexes = holdOut(N, validationRatio)
                newTrainingInputs = trainingInputs[trainingIndexes, :]
                validationInputs = trainingInputs[validationIndexes, :]
                newTrainingTargets = trainingTargets[trainingIndexes, :]
                validationTargets = trainingTargets[validationIndexes, :]
                ann, _... = trainClassANN(topology, (newTrainingInputs, newTrainingTargets);
                validationDataset = (validationInputs, validationTargets), testDataset = (testInputs, testTargets),
                transferFunctions = transferFunctions, maxEpochs = maxEpochs, maxEpochsVal = maxEpochsVal, minLoss = minLoss,
                learningRate = learningRate)
            else
                ann, _... = trainClassANN(topology, (trainingInputs, trainingTargets);
                transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss,
                learningRate = learningRate)
            end
            testOutputs = ann(testInputs')
            accuracyExecution[execution], errorRateExecution[execution], sensitivityExecution[execution],
            specificityExecution[execution], ppvExecution[execution], npvExecution[execution],
            f1ScoreExecution[execution] = confusionMatrix(testOutputs', testTargets)
        end
        accuracy[fold] = mean(accuracyExecution)
        errorRate[fold] = mean(errorRateExecution)
        sensitivity[fold] = mean(sensitivityExecution)
        specificity[fold] = mean(specificityExecution)
        ppv[fold] = mean(ppvExecution)
        npv[fold] = mean(npvExecution)
        f1Score[fold] = mean(f1ScoreExecution)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)),
    (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
    (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1Score), std(f1Score)))
end;