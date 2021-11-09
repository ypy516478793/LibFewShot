cd ../

GPU="6,7"

python run_trainer.py \
    -c "./config/baseline.yaml" -n_gpu 2 -gpus $GPU

python run_trainer.py \
    -c "./config/baseline++.yaml" -n_gpu 2 -gpus $GPU

python run_trainer.py \
    -c "./config/rfs.yaml" -n_gpu 2 -gpus $GPU

python run_trainer.py \
    -c "./config/skd.yaml" -n_gpu 2 -gpus $GPU


#python run_test.py \
#    -c "./results/RelationNet-miniImageNet--ravi-Conv64F-5-1/config.yaml" \
#    -n_gpu 2 -gpus $GPU
