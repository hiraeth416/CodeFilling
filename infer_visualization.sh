CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_single --fusion_method no --comm_thre=0  --result_name 'egocamera_otherlidar' --modal 3 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_single --fusion_method no --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_single --fusion_method no --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'egocamera_otherlidar' --modal 3 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"

CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_hmvit_new --fusion_method intermediate --comm_thre=0  --result_name 'egocamera_otherlidar' --modal 3 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_hmvit_new --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/Dairv2x_hmvit_new --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"

exp_id='codebook_dairv2x_size4'
comm_thre=(0 0.01 0.10 0.20 0.60 1)

for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'lidar_only'  --comm_thre=$i --note 'new_comm_lidar'$i  --modal 0  --range "102.4,48"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'camera_only' --comm_thre=$i --note 'new_comm_camera'$i  --modal 1  --range "102.4,48"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id --fusion_method intermediate --result_name 'egocamera_otherlidar' --comm_thre=$i --note 'new_comm_egocamera'$i  --modal 3 --range  "102.4,48"
done


exp_id1='Dairv2x_where2comm'
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id1 --fusion_method intermediate --result_name 'lidar_only'  --comm_thre=$i --note 'new_comm_lidar'$i  --modal 0  --range "102.4,48"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id1 --fusion_method intermediate --result_name 'camera_only' --comm_thre=$i --note 'new_comm_camera'$i  --modal 1  --range "102.4,48"
done
for i in ${comm_thre[*]}
do
    echo $i
    CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference_for_visualization.py --model_dir '/GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/'$exp_id1 --fusion_method intermediate --result_name 'egocamera_otherlidar' --comm_thre=$i --note 'new_comm_egocamera'$i  --modal 3 --range  "102.4,48"
done