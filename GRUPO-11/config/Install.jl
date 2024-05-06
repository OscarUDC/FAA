# Package Install
begin
    import Pkg;
    Pkg.add("XLSX");
    Pkg.add("FileIO");
    Pkg.add("JLD2");
    Pkg.add("Flux");
    Pkg.add("ScikitLearn");
    Pkg.add("Plots");
    Pkg.add("MAT");
    Pkg.add("Images");
    Pkg.add("DelimitedFiles");
    Pkg.add("CSV");
end

# Read Iris Dataset
using DelimitedFiles
dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
targets = dataset[:,5];

print(typeof(dataset))
print(typeof(inputs))
print(typeof(targets))

inputs = convert(Array{Float64,2},inputs);
targets = convert(Array{String,1},targets);

print(typeof(inputs))
print(typeof(targets))
