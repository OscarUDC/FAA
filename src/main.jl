# Charge the modules
include.([
    "01_oneHotEncoding.jl",
    "02_normalization.jl",
    "03_classification.jl",
    "04_accuracy.jl",
    "05_neuralnetwork_construction.jl",
    "06_neuralnetwork_training.jl"
])


# Using the functions
data = # Your dataset
classes = # Your classes

# Example usage
encoded_data = OneHotEncoding.oneHotEncoding(data, classes)
normalized_data = Normalization.normalizeMinMax(encoded_data)
outputs = NeuralNetworkConstruction.buildClassANN(...) |> NeuralNetworkTraining.trainClassANN(...)

# Other operations and logic in your program
