using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    validTypes = [:ANN, :SVC, :DecisionTreeClassifier, :KNeighborsClassifier]
    if !(modelType in validTypes)
        throw(ArgumentError("Model type $modelType does not exist or is not allowed"))
    end
    newModelHyperparameters = Dict()
    for (key, value) in modelHyperparameters
        newModelHyperparameters[Symbol(key)] = value
    end
    if modelType == :ANN
        return ANNCrossValidation(;inputs = inputs, targets = targets, crossValidationIndices = crossValidationIndices, newModelHyperparameters...)
    end
    model = eval(modelType)(;newModelHyperparameters) 
    targets = string.(targets)
    numFolds = maximum(crossValidationIndices)
    inputs = confusionMatrix(inputs, targets)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = targets[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = targets[crossValidationIndices .== fold, :]
        trainedModel, _... = fit!(model, trainingInputs, trainingTargets)
        testOutputs = trainedModel(testInputs')
        accuracy[fold], errorRate[fold], sensitivity[fold], specificity[fold], ppv[fold], npv[fold], f1Score[fold] = confusionMatrix(testOutputs', testTargets)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)), (mean(sensitivity), std(sensitivity)),
    (mean(specificity), std(specificity)), (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1Score), std(f1Score)))
end;