using ScikitLearn
@sk_import svm: SVC 
@sk_import tree: DecisionTreeClassifier 
@sk_import neighbors: KNeighborsClassifier

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, 
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, 
    crossValidationIndices::Array{Int64,1})
end;