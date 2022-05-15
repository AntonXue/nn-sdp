import Pkg

# Make sure MOSEK license
@assert isfile(joinpath(homedir(), "mosek", "mosek.lic"))

# Install a bunch of packages
Pkg.add("ArgParse")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Dualization")
Pkg.add("JuMP")
Pkg.add("MosekTools")
Pkg.add("NaturalSort")
Pkg.add("Parameters")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("Reexport")

# Some Conda-specific installations
Pkg.add("Conda")
import Conda
Conda.add("numpy")
Conda.add("appdirs")
Conda.add("packaging")
Conda.add("pytorch", channel="pytorch")
Conda.add("torchvision", channel="pytorch")
Conda.add("onnx")
Conda.add("onnxruntime", channel="conda-forge")
Conda.add("mosek", channel="mosek")

# Set up PyCall and have it point to Conda
Pkg.add("PyCall")
ENV["PYTHON"] = ""
Pkg.build("PyCall")

# Some pip-specific installations
import PyCall
run(`$(PyCall.python) -m pip install git+https://github.com/AntonXue/onnx2pytorch.git`)


