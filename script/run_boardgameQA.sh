export PYTHONPATH="$PWD" 
python3 src/rl/curriculum.py --num_gpus 8 --gpu_memory 40 --task_name boardgameQA \
    --cuda_visible_devices 0,1,2,3,4,5,6,7 --Lambda 0.2 --n_samples 16