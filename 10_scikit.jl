using ScikitLearn: @sk_import, fit!, predict

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    if modelType == :ANN
        if !haskey(modelHyperparameters, "numExecutions")
            modelHyperparameters["numExecutions"] = 50
        end
        if !haskey(modelHyperparameters, "transferFunctions")
            modelHyperparameters["transferFunctions"] = fill(σ, length(modelHyperparameters["topology"]))
        end
        if !haskey(modelHyperparameters, "maxEpochs")
            modelHyperparameters["maxEpochs"] = 1000
        end
        if !haskey(modelHyperparameters, "minLoss")
            modelHyperparameters["minLoss"] = 0.0
        end
        if !haskey(modelHyperparameters, "learningRate")
            modelHyperparameters["learningRate"] = 0.01
        end
        if !haskey(modelHyperparameters, "validationRatio")
            modelHyperparameters["validationRatio"] = 0
        end
        if !haskey(modelHyperparameters, "maxEpochsVal")
            modelHyperparameters["maxEpochsVal"] = 20
        end
        if !haskey(modelHyperparameters, "showText")
            modelHyperparameters["showText"] = false
        end
        return ANNCrossValidation(modelHyperparameters["topology"], inputs, targets, crossValidationIndices;
        numExecutions = modelHyperparameters["numExecutions"],
        transferFunctions = modelHyperparameters["transferFunctions"],
        maxEpochs = modelHyperparameters["maxEpochs"], minLoss = modelHyperparameters["minLoss"],
        learningRate = modelHyperparameters["learningRate"], validationRatio = modelHyperparameters["validationRatio"],
        maxEpochsVal = modelHyperparameters["maxEpochsVal"], showText = modelHyperparameters["showText"])
    end 
    targets = string.(targets)
    if modelType == :SVC
        if !haskey(modelHyperparameters, "kernel")
            modelHyperparameters["kernel"] = "rbf"
        end
        if !haskey(modelHyperparameters, "degree")
            modelHyperparameters["degree"] = 3
        end
        if !haskey(modelHyperparameters, "gamma")
            modelHyperparameters["gamma"] = "scale"
        end
        if !haskey(modelHyperparameters, "coef0")
            modelHyperparameters["coef0"] = 0.0
        end
        model = SVC(C = modelHyperparameters["C"], kernel = modelHyperparameters["kernel"],
        degree = modelHyperparameters["degree"], gamma = modelHyperparameters["gamma"],
        coef0 = modelHyperparameters["coef0"])
    elseif modelType == :DecisionTreeClassifier
        if !haskey(modelHyperparameters, "max_depth")
            modelHyperparameters["max_depth"] = nothing
        end
        model = DecisionTreeClassifier(max_depth = modelHyperparameters["max_depth"], random_state = 1)
    elseif modelType == :KNeighborsClassifier
        if !haskey(modelHyperparameters, "n_neighbors")
            modelHyperparameters["n_neighbors"] = 5
        end
        model = KNeighborsClassifier(n_neighbors = modelHyperparameters["n_neighbors"])
    else
        throw(ArgumentError("Model type $modelType does not exist or is not allowed"))
    end
    numFolds = maximum(crossValidationIndices)
    inputs = confusionMatrix(inputs, targets)
    (accuracy, errorRate, sensitivity, specificity, ppv, npv, f1Score) = (Vector{Float64}(undef, numFolds) for _ in 1:7)
    for fold in 1:numFolds
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        trainingTargets = targets[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        testTargets = targets[crossValidationIndices .== fold, :]
        trainedModel = fit!(model, trainingInputs, trainingTargets)
        testOutputs = predict(trainedModel, testInputs)
        accuracy[fold], errorRate[fold], sensitivity[fold], specificity[fold],
        ppv[fold], npv[fold], f1Score[fold] = confusionMatrix(testOutputs, testTargets)
    end
    return ((mean(accuracy), std(accuracy)), (mean(errorRate), std(errorRate)), (mean(sensitivity), std(sensitivity)),
    (mean(specificity), std(specificity)), (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1Score), std(f1Score)))
end;