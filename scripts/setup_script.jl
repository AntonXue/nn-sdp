import Pkg

# Make sure MOSEK license
mosek_lic_exists = isfile(joinpath(homedir(), "mosek", "mosek.lic"))
if !mosek_lic_exists
  error("~/mosek/mosek.lic does not exist")
end

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

### Set up the directories

# Make some directories if they don't yet exist
BENCH_DIR = joinpath(@__DIR__, "..", "bench")
BENCH_ACAS_DIR = joinpath(@__DIR__, "..", "bench", "acas")
BENCH_RAND_DIR = joinpath(@__DIR__, "..", "bench", "rand")
mkpath(BENCH_ACAS_DIR)
mkpath(BENCH_RAND_DIR)

DUMP_DIR = joinpath(@__DIR__, "..", "dump")
DUMP_SPARSITY_DIR = joinpath(@__DIR__, "..", "dump", "sparsity")
DUMP_ACAS_DIR = joinpath(@__DIR__, "..", "dump", "acas")
DUMP_REACH_DIR = joinpath(@__DIR__, "..", "dump", "reach")
DUMP_SCALE_DIR = joinpath(@__DIR__, "..", "dump", "scale")
mkpath(DUMP_SPARSITY_DIR)
mkpath(DUMP_ACAS_DIR)
mkpath(DUMP_REACH_DIR)
mkpath(DUMP_SCALE_DIR)


