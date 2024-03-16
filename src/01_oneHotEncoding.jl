# Funciones para codificar entradas y salidas categóricas

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    num_patterns = length(feature)

    if num_classes == 2
        encoded_matrix = reshape(feature .== classes[1], :, 1)
    else
        encoded_matrix = falses(num_patterns, num_classes)

        for i in 1:num_classes
            encoded_matrix[:, i] .= feature .== classes[i]
        end
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
