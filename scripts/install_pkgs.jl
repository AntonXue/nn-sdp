import Pkg

# Make sure MOSEK license
@assert isfile(joinpath(homedir(), "mosek", "mosek.lic"))

# Install a bunch of packages
Pkg.add("ArgParse")
Pkg.add("Parameters")
Pkg.add("Reexport")
Pkg.add("JuMP")
Pkg.add("Dualization")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("MosekTools")
Pkg.add("DataFrames")
Pkg.add("CSV")

# Some Conda-specific installations
Pkg.add("Conda")
import Conda
Conda.add("appdirs")
Conda.add("pytorch", channel="pytorch")
Conda.add("torchvision", channel="pytorch")
Conda.add("onnx")
Conda.add("onnxruntime", channel="conda-forge")

# Set up PyCall and have it point to Conda
Pkg.add("PyCall")
ENV["PYTHON"] = ""
Pkg.build("PyCall")

