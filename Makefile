# Simulation runs. Usage: make <target> [VAR=value ...], e.g. make ff T=2000 NAME=ff_short
# Each run writes a folder in $(EXPERIMENT_PATH). The first run compiles the C++ model (a few minutes).

PY ?= python3
T ?= 3500
NAME ?= $@
STATE ?= spont
PERT ?= L23E
T_PERT ?= 2000
STATES ?= spont,FF,FB,FF_FB
WORKERS ?= 8
export EXPERIMENT_PATH ?= runs
RUN ?= $(shell ls -td $(EXPERIMENT_PATH)/run_* 2>/dev/null | head -1)

SIM = $(PY) simulate.py t_sim=$(T) experiment_type=$(NAME)

.PHONY: help env spont ff fb ff-fb fb-states perturb matrices matrices-fb density plots clean-build

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

env: ## Create the conda environment
	conda env create -f environment.yml

spont: ## Spontaneous activity (Fig 2)
	$(SIM)

ff: ## Feedforward input from 700 ms (Fig 1C)
	$(SIM) input.state=FF input.t_state=700

fb: ## Feedback input to L5 from 700 ms (Fig 1D)
	$(SIM) input.state=FB input.t_state=700

ff-fb: ## Feedforward + feedback from 700 ms (Fig 6A)
	$(SIM) input.state=FF_FB input.t_state=700

fb-states: ## The four feedback states of Fig 5A, one run each
	for s in FB23 FB23_5 FB6 FB23_5_6; do $(PY) simulate.py t_sim=$(T) experiment_type=$$s input.state=$$s input.t_state=700 || exit 1; done

perturb: ## One perturbation: 30 pA to PERT from T_PERT in state STATE
	$(SIM) input.state=$(STATE) input.t_state=0 input.pert=$(PERT) input.t_pert=$(T_PERT)

matrices: ## Perturbation matrices for STATES (Figs 3, 4, 6)
	$(PY) perturbation_matrix.py experiment_type=$(NAME) 'sweep.states=[$(STATES)]' sweep.workers=$(WORKERS)

matrices-fb: ## Perturbation matrices of the four feedback states (Fig 5B)
	$(MAKE) matrices STATES=FB23,FB23_5,FB6,FB23_5_6 NAME=$@

plots: ## Paper-style figures of run folder RUN (default: latest) into RUN/figures
	$(PY) plots.py $(RUN)

density: ## Synaptic density sweep (run_simulations.sh)
	bash run_simulations.sh

clean-build: ## Delete the compiled C++ model (forces a full rebuild)
	rm -rf standalone_build
