#Importamos los módulos utilizados

using Random
include("32740547R_32915493D_32727069R.jl");

#Parametros auxiliares
"""
its a generator of topologies,
inputs
===========
number_of_topologies
number_of_layers
min_neurons_per_layer
max_neurons_per_layer
num_inputs
num_outputs
===========
outputs
===========
an array of arrays, topologies, which every array is a topology
"""
function CreationOfTopologies(number_of_topologies, number_of_layers, min_neurons_per_layer, max_neurons_per_layer, num_inputs, num_outputs)
    topologies = []
    for topology in number_of_topologies
        this_topology = []
        push!(this_topology, num_inputs)
        for layer in number_of_layers
            push!(this_topology, rand(min_neurons_per_layer:max_neurons_per_layer))
        end
        if num_outputs == 2
            push!(this_topology, 1)
        else
            push!(this_topology, num_outputs)
        end
        push!(topologies, this_topology)
    end
    return topologies
end

using DelimitedFiles
dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
targets = dataset[:,5];

ANN_1 = trainClassANN(topology1, dataset);
ANN_2 = trainClassANN(topology2, dataset);
ANN_3 = trainClassANN(topology3, dataset);
ANN_4 = trainClassANN(topology4, dataset);
ANN_5 = trainClassANN(topology5, dataset);
ANN_6 = trainClassANN(topology6, dataset);
ANN_7 = trainClassANN(topology7, dataset);
ANN_8 = trainClassANN(topology8, dataset);