#4.1 and 4.2
using LinearAlgebra
include.([
    "04_accuracy.jl",
    "01_oneHotEncoding.jl"
    ])

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    VP = sum(outputs .& targets)            # Verdaderos positivos
    VN = sum((.!outputs) .& (.!targets))    # Verdaderos negativos
    FP = sum(outputs .& .!targets)  # Falsos positivos
    FN = sum(.!outputs .& targets)  # Falsos negativos

    if (VP + VN + FP + FN) == 0
        accuracy = 0.0
    else
        accuracy = (VN + VP) / (VN + VP + FN + FP)
    end
    
    if (VN + VP + FN + FP) == 0
        error_rate = 0.0
    else
        error_rate = (FN + FP) / (VN + VP + FN + FP)
    end

#-----------------------
    
    if VP == 0 && FN == 0
        sensitivity = 1.0
    else
        sensitivity = VP / (FN + VP)
    end
    
    if VN == 0 && FP == 0
        specificity = 1.0
    else
        specificity = VN / (FP + VN)
    end
    
    if VP == 0 && FP == 0
        ppv = 1.0
    else
        ppv = VP / (VP + FP)
    end

    if VN == 0 && FN == 0
        npv = 1.0
    else
        npv = VN / (VN + FN)
    end
    
    if sensitivity == 0 && prec == 0
        f1_score = 0.0
    else
        f1_score = 2 * (accuracy * sensitivity) / (accuracy + sensitivity)
    end

    confusion_matrix = [VP FN; FP VN]

    return (accuracy, error_rate, sensitivity, specificity, ppv, npv, f1_score, confusion_matrix)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    binary_outputs = outputs .≥ threshold
    return confusionMatrix(binary_outputs, targets) #? Cogemos solo los outputs que pasen el umbral
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



# # Generar conjuntos de datos aleatorios
# using Random
# Random.seed!(123)  # Fijar semilla para reproducibilidad

# # Generar datos de prueba
# outputs = rand(100)
# targets = rand(Bool, 100)

# # Calcular la matriz de confusión con diferentes umbrales
# resultados_threshold_05 = confusionMatrix(outputs, targets; threshold=0.5)
# resultados_threshold_06 = confusionMatrix(outputs, targets; threshold=0.6)

# # Validar los resultados
# println("Resultados con umbral 0.5:", resultados_threshold_05)
# println("Resultados con umbral 0.6:", resultados_threshold_06)

#-----------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    n_classes = size(outputs, 2)
    n_patterns = size(outputs, 1)
    if n_classes <= 2 || n_patterns <= 2
        error("Invalid input dimensions")
    end
    
    # Verificar si solo hay una columna y llamar a la función anterior
    if n_classes == 1
        return confusionMatrix(outputs[:, 1], targets[:, 1]; weighted=weighted)
    end

    # Inicialización de vectores para métricas
    sensitivity = zeros(Float64, n_classes)
    specificity = zeros(Float64, n_classes)
    VPP = zeros(Float64, n_classes)
    VPN = zeros(Float64, n_classes)
    F1 = zeros(Float64, n_classes)

    # Cálculo de métricas para cada clase
    for i in 1:n_classes
        outputs_class = outputs[:, i]
        targets_class = targets[:, i]
        class_metrics = confusionMatrix(outputs_class, targets_class)
        sensitivity[i] = class_metrics[3]
        specificity[i] = class_metrics[4]
        VPP[i] = class_metrics[5]
        VPN[i] = class_metrics[6]
        F1[i] = class_metrics[7]
    end

    # Cálculo de métricas macro o weighted
    if weighted
        weights = sum(targets, dims=1)
        prec = dot(VPP, weights) / sum(weights)
        error_rate = 1 - prec
        sensitivity = dot(sensitivity, weights) / sum(weights)
        specificity = dot(specificity, weights) / sum(weights)
        VPP = dot(VPP, weights) / sum(weights)
        VPN = dot(VPN, weights) / sum(weights)
        F1 = dot(F1, weights) / sum(weights)
    else
        prec = accuracy(outputs, targets)
        error_rate = 1 - prec
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        VPP = mean(VPP)
        VPN = mean(VPN)
        F1 = mean(F1)
    end

    # Cálculo de matriz de confusión
    confusion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:n_classes, j in 1:n_classes]

    return (prec, error_rate, sensitivity, specificity, VPP, VPN, F1, confusion_matrix)
end


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    outputs_bool = classifyOutputs(outputs)
    return confusionMatrix(outputs_bool, targets; weighted=weighted)
end;


function confusionMatrix(outputs::AbstractVector{<:Any}, targets::AbstractVector{<:Any}; weighted::Bool=true)

    # Asegurar que todos los elementos de outputs estén presentes en targets
    @assert all([in(output, unique(targets)) for output in outputs]) "All elements of outputs must be present in targets"
    
    # Convertir los vectores targets y outputs a matrices one-hot
    targets_onehot = oneHotEncoding(targets)
    outputs_onehot = oneHotEncoding(outputs)
    
    # Llamar a la función confusionMatrix con las matrices one-hot
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end;



function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Calcular la matriz de confusión
    confusion_matrix = confusionMatrix(outputs, targets; weighted=weighted)[end]
    
    # Imprimir la matriz de confusión
    println("Confusion Matrix:")
    for i in axes(confusion_matrix, 1)
        println(confusion_matrix[i, :])
    end
end;


# Importar paquetes necesarios
using Random

# Definir función de generación de datos de ejemplo
function generateExampleData(n_samples::Int, n_classes::Int)
    outputs = rand(Bool, n_samples, n_classes)
    targets = rand(Bool, n_samples, n_classes)
    return outputs, targets
end

# Generar datos de ejemplo
n_samples = 100
n_classes = 3
outputs, targets = generateExampleData(n_samples, n_classes)

# Probar la función confusionMatrix
prec, error_rate, sensitivity, specificity, VPP, VPN, F1, confusion_matrix = confusionMatrix(outputs, targets)
println("Metrics:")
println("prec: ", prec)
println("Error Rate: ", error_rate)
println("Sensitivity: ", sensitivity)
println("Specificity: ", specificity)
println("VPP: ", VPP)
println("VPN: ", VPN)
println("F1: ", F1)
println()

# Probar la función printConfusionMatrix
printConfusionMatrix(outputs, targets)
