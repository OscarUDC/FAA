#4.1 and 4.2
using Statistics

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets)
    VN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    accuracy = (VP + VN + FP + FN) == 0 ? 0.0 : (VN + VP) / (VP + VN + FP + FN)
    error_rate = (VP + VN + FP + FN) == 0 ? 0.0 : (FN + FP) / (VP + VN + FP + FN)
    sensitivity = (VP + FN) == 0 ? 1.0 : VP / (FN + VP)
    specificity = (VN + FP) == 0 ? 1.0 : VN / (FP + VN)
    ppv = (VP + FP) == 0 ? 1.0 : VP / (VP + FP)
    npv = (VN + FN) == 0 ? 1.0 : VN / (VN + FN)
    f1_score = (sensitivity + ppv) == 0 ? 0.0 : 2 * (ppv * sensitivity) / (ppv + sensitivity)
    confusion_matrix = [VN FP; FN VP]
    return (accuracy, error_rate, sensitivity, specificity, ppv, npv, f1_score, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    return confusionMatrix(binary_outputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    results = confusionMatrix(outputs, targets)
    
    println("accuracy: ", results[1])
    println("Error Rate: ", results[2])
    println("Sensitivity: ", results[3])
    println("Specificity: ", results[4])
    println("PPV: ", results[5])
    println("NPV: ", results[6])
    println("F1-Score: ", results[7])
    println("Confusion Matrix:")
    println(results[8])
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold #? Si el número de outpost es mayor que el umbral, hacemos un print de la matriz de confusión
    printConfusionMatrix(binary_outputs, targets)
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    _, n_classes = size(outputs)  # Obtener las dimensiones de las matrices
    if n_classes != size(targets, 2)  # Verificar si el número de columnas es diferente en las matrices outputs y targets
        error("Number of columns in outputs and targets matrices must be equal")
    end

    if n_classes == 1  # Si solo hay una clase, llamar a la función anterior
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end
    # Inicializar vectores de métricas con ceros
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)

    # Calcular matriz de confusión(sin necesidad de inicializarla primero con ceros)
    _, _, sensitivity[i], specificity[i], VPP[i], VPN[i], F1[i], _ = [sum(outputs[:, i] .& targets[:, i]) for i in axes(outputs, 2)]

    accuracy = accuracy(outputs, targets)
    confusionMatrix = zeros(n_classes, n_classes)
    for row in eachindex(outputs, 1)
        realClass = findfirst(targets[row, :])
        predictedClass = findfirst(outputs[row, :])
        confusionMatrix[realClass, predictedClass] += 1
    end
    if !weighted
        return(accuracy, 1 - accuracy, mean(sensitivity), mean(specificity), mean(VPP), mean(VPN), mean(F1), confusionMatrix)
    else
        ponderation = sum(confusionMatrix, dims = 2) / size(targets, 1)
        sensitivity *= ponderation
        specificity *= ponderation
        VPP *= ponderation
        VPN *= ponderation
        F1 *= ponderation
    return (accuracy, error_rate, sum(sensitivity), sum(specificity), sum(VPP), sum(VPN), sum(F1), confusion_matrix)
    end
end;

# Definir función confusionMatrix para matrices de valores reales
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end;

# Definir función confusionMatrix para vectores de cualquier tipo
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurarse de que todas las clases del vector de outputs estén incluidas en el vector de targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end;


function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    metrics = confusionMatrix(outputs, targets; weighted=weighted)
    println("Metrics:")
    println("Precision: ", metrics[1])
    println("Error rate: ", metrics[2])
    println("Sensitivity: ", metrics[3])
    println("Specificity: ", metrics[4])
    println("VPP: ", metrics[5])
    println("VPN: ", metrics[6])
    println("F1: ", metrics[7])
    println("Confusion matrix:")
    println(metrics[8])
    return metrics
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return printConfusionMatrix(outputs_bool, targets; weighted=weighted)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Asegurar que todos los elementos de outputs estén presentes en targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    metrics = confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
    println("Metrics:")
    println("Precision: ", metrics[1])
    println("Error rate: ", metrics[2])
    println("Sensitivity: ", metrics[3])
    println("Specificity: ", metrics[4])
    println("VPP: ", metrics[5])
    println("VPN: ", metrics[6])
    println("F1: ", metrics[7])
    println("Confusion matrix:")
    println(metrics[8])
    return metrics
end;