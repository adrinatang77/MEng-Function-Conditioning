# for i in {0..144}; do
#     python -m scripts.annotate_foldcomp --db /scratch/projects/cgai/openprot/data/afdb_rep_v4/afdb_rep_v4 --out tmp/$i.pkl --num_workers 144 --worker_id $i & 
# done

# for i in {0..35}; do
for i in {0..31}; do
    # python -m scripts.annotate_foldcomp --db /data/cb/scratch/datasets/afdb_uniref_v4/afdb_uniprot_v4 --out tmp/afdb_uniprot_v4.idx.$i --num_workers 72 --worker_id $i & 
    # python -m scripts.foldcomp_nma --db /data/cb/scratch/datasets/afdb_uniref_v4/afdb_uniprot_v4 --out tmp/nma/afdb_uniprot_v4.idx.$i --num_workers 72 --worker_id $i & 
    
    python -m scripts.unpack_alphafill --num_workers 32 --worker_id $i & 
done
wait