# nn-sdp

This repository contains an implementation of Chordal-DeepSDP, a chordally sparse formulation of the DeepSDP framework for safety verification of feedforward neural newtorks.

# Requirements and Installation
This codebase was tested with the following:
* Julia 1.7.2
* MOSEK 9.3
* Python 3.9

It is assumed that Julia is in your executable path. To install the relevant libraries, run the following (note: this will change your Conda.jl settings)
```
julia scripts/setup_script.jl
```

# Running Experiments
The relevant neural network files are present in `bench/rand`.

To rerun the scalability experiments, run
```
julia -i experiments/scale.jl
```
which will dump the relevant CSV files into `dump/scale`.

To run the reachability experiments, run
```
julia -i experiments/reach.jl
```
whichwill the dump the relevant images into `dump/reach`.
