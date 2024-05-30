# Official Version
gpu_id=$2
# exp_id=$2
# result_name=$3 

# exp_id='opv2v_point_pillar_lidar_where2comm_kalman_ms_tune_allo_2023_05_22_15_01_29'
# exp_id='HEAL_opv2v_m1_pointpillar_m2_lsseff_end2end_140.8_40_2023_07_05_21_30_17'

exp_id='Dairv2x_v2xvit_new'
# comm_thre=(0 0.01 0.001 1.0 0.20)
# comm_thre=(0.003 0.03 0.10)
# comm_thre=(0.30 0.40 0.60 0.80)
# comm_thre=(0.30)
# comm_thre=(0.20 0.40 0.60)
#comm_thre=(0 0.001 0.003 0.005 0.007 0.01 0.03 0.05 0.07 0.10 0.20 0.40 0.60 1)
comm_thre=(0.0)

for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'egocamera_otherlidar' --comm_thre=$i --note 'new_comm_egocamera'$i  --modal 3 --range  "102.4,51.2"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'lidar_only'  --comm_thre=$i --note 'new_comm_lidar'$i  --modal 0  --range "102.4,51.2"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'camera_only' --comm_thre=$i --note 'new_comm_camera'$i  --modal 1  --range "102.4,51.2"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'egorandom_ratio0.5' --comm_thre=$i --note 'egorandom_ratio0.5'$i  --modal 4 --range  "102.4,51.2"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'egolidar_othercamera' --comm_thre=$i --note 'new_comm_egolidar'$i  --modal 2 --range  "102.4,51.2"
done