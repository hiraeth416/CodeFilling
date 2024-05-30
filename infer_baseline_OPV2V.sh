#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_Attfuse --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_Attfuse --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_Attfuse --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,48"

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_CoBEVT --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_CoBEVT --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_CoBEVT --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,48"

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,48"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_DiscoNet --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,48"

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_HMVIT --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,51.2"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_HMVIT --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,51.2"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_HMVIT --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,51.2"

CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2VNet --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,102.4"
CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2VNet --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,102.4"
CUDA_VISIBLE_DEVICES=5 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2VNet --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,102.4"

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2X-ViT --fusion_method intermediate --comm_thre=0  --result_name 'lidar_only' --modal 0 --range  "102.4,51.2"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2X-ViT --fusion_method intermediate --comm_thre=0  --result_name 'camera_only' --modal 1 --range  "102.4,51.2"
#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference.py --model_dir  /GPFS/rhome/sifeiliu/OpenCOODv2_new/opencood/logs_HEAL/OPV2V_V2X-ViT --fusion_method intermediate --comm_thre=0  --result_name 'egorandom_ratio0.5' --modal 4 --range  "102.4,51.2"
