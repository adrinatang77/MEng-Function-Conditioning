for i in {0..144}; do
    python -m scripts.annotate_foldcomp --db /scratch/projects/cgai/openprot/data/afdb_rep_v4/afdb_rep_v4 --out tmp/$i.pkl --num_workers 144 --worker_id $i & 
done