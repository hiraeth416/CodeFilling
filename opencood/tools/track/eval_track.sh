model_dir=$1
model_name=$2
note=$3

# python opencood/tools/track/sort.py --det_logs_path $model_dir'/npy'

python opencood/tools/track/AB3DMOT.py --det_logs_path $model_dir'/npy'

python opencood/tools/track/eval_mot.py --model_dir $model_dir --model_name $model_name --note $note
