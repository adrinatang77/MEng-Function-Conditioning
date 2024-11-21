EVAL_DIR=$SAMPLE_DIR/eval${START}_${END}
ROOT=/home1/10165/bjing/work/insilico_design_pipeline
eval "$(conda shell.bash hook)"
conda activate eval

$@