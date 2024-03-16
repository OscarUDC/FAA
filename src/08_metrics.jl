function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    TP = sum(outputs .& targets)
    TN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& (.!targets))
    FN = sum((.!outputs) .& targets)

    precision = TP / (TP + FP)
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = precision
    npv = TN / (TN + FN)
    f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

    confusion_matrix = [TP FN; FP TN]

    return (precision, error_rate, sensitivity, specificity, ppv, npv, f1_score, confusion_matrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    return confusionMatrix(binary_outputs, targets)
end

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    results = confusionMatrix(outputs, targets)
    
    println("Precision: ", results[1])
    println("Error Rate: ", results[2])
    println("Sensitivity: ", results[3])
    println("Specificity: ", results[4])
    println("PPV: ", results[5])
    println("NPV: ", results[6])
    println("F1-Score: ", results[7])
    println("Confusion Matrix:")
    println(results[8])
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    printConfusionMatrix(binary_outputs, targets)
end



