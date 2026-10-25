# V2 launch environment. Sourcing this file does not select a GPU or start work.
export HETEROVLA_ROOT="${HETEROVLA_ROOT:-$HOME/heterovla}"
export OPENPI_ROOT="${OPENPI_ROOT:-$HETEROVLA_ROOT/third_party/openpi}"
export HETEROVLA_DATA_ROOT="${HETEROVLA_V2_DATA_ROOT:-$HOME/heterovla-data/aligned/V2}"
export HETEROVLA_RUN_ROOT="${HETEROVLA_V2_RUN_ROOT:-$HOME/heterovla-model/V2/runs}"
export PI05_PARAMS="${HETEROVLA_PI05_BASE_PARAMS:-$HOME/heterovla-model/pretrained/pi05_base/params}"
export OPENPI_DATA_HOME="${HETEROVLA_V2_OPENPI_CACHE:-$HOME/heterovla-model/V2/cache/openpi}"
export HF_LEROBOT_HOME="$HETEROVLA_DATA_ROOT/lerobot"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_PROJECT=heterovla-pi05
export WANDB_GROUP=pi05-v2-full-20k
export WANDB_DIR="${HETEROVLA_V2_WANDB_DIR:-$HOME/heterovla-model/V2/wandb}"
export WANDB_DISABLE_CODE=true
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${HETEROVLA_V2_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$HETEROVLA_ROOT/train/V2/scripts:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

