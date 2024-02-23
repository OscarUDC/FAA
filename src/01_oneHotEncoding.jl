# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    unique_classes = unique(classes)                                        # Nos devuelve un array con las clases que existen en el db
    encoded_matrix = zeros(Int, length(feature), length(unique_classes))    # Inicializamos una matriz de ceros para almacenar la codificación one-hot

    for i in 1:length(feature)                                              # Iteramos sobre cada patrón en el vector de características
        class_index = findfirst(x -> x == classes[i], unique_classes)       # Encontramos el índice de la clase correspondiente en el array de clases únicas
        encoded_matrix[i, class_index] = 1                                  # Marcamos la posición correspondiente a la clase con un 1 en la matriz one-hot
    end

    return encoded_matrix
end;

# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)                                              # Obtenemos las clases únicas presentes en el vector de características
    return oneHotEncoding(feature, classes)                                # Llamamos al método con la información de clases obtenida

end;

# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return hcat(feature)                                                   # Convertimos el vector booleano en una matriz columna

end;

# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente
