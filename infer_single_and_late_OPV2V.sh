#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_old.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_late --fusion_method late --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_old.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_late --fusion_method late --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"
CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference_old.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_late --fusion_method late --comm_thre=0  --result_name 'egocamera_otherlidar' --modal 3 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_single --fusion_method no --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_single --fusion_method no --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_single --fusion_method no --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2/opencood/logs/opv2v_lss_single_72_48_2023_07_04_11_58_22 --fusion_method late --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"