using Flux

include.("32740547R_32915493D_32727069R.jl")

# Las observaciones están a lo largo de las filas y las características a lo largo de las columnas
inputs = rand(Float32, 1000, 5)  # 100 muestras x 10 características
targets = inputs .> mean(inputs)  # 100 muestras, datos binarios de objetivo

dataset = (inputs, targets)
# Definir la topología de la red (número de neuronas en cada capa)
topology = [32, 16, 16, 16, 2]
# Llamar a la función trainClassANN
# TARDA UN RATO, REDUCIR maxEpochs!!!!
model = trainClassANN(topology, dataset; maxEpochs = 100000, learningRate = 0.05)
