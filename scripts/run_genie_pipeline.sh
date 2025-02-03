SAMPLE_DIR=$(pwd)/$1
START=$2
END=$3

EVAL_DIR=$SAMPLE_DIR/eval${START}_${END}
# ROOT=/home1/10165/bjing/work/insilico_design_pipeline
ROOT=../insilico_design_pipeline
eval "$(conda shell.bash hook)"
conda activate eval

mkdir -p $EVAL_DIR/pdbs
for i in $(seq $START $END); do
    cp $SAMPLE_DIR/sample$i.pdb $EVAL_DIR/pdbs
done
cd $ROOT

python pipeline/standard/evaluate.py --rootdir $EVAL_DIR --version unconditional
cat $EVAL_DIR/info.csv
echo $EVAL_DIR/info.csv

