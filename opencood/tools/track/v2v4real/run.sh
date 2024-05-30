d=3

# # No
# dir=''
# # Late
# dir=''
# # Early
# dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_early_2023_06_21_19_57_47'
# # F-Cooper
# dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_fcooper_2023_06_21_07_51_44'
# # V2VNet
dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_v2vnet_2023_06_28_01_47_15'
# # V2XViT
# dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_v2xvit_2023_06_21_07_55_11'
# # where2comm(max)
# dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_multiscale_max_2023_06_21_07_51_38'
# # where2comm(attn)
# dir='/remote-home/share/sizhewei/logs_v2/v2v4real_point_pillar_lidar_multiscale_att_2023_06_28_09_32_06'

# opencood/tools/track/opv2v/0.sh
# chmod +x opencood/tools/track/opv2v/0.sh
#clear

# CUDA_VISIBLE_DEVICES=$d python opencood/tools/inference_track.py --model_dir "$dir/" --save_track

# CUDA_VISIBLE_DEVICES=$d python opencood/tools/track/AB3DMOT.py --det_logs_path "$dir/intermediate_epoch27/npy"

CUDA_VISIBLE_DEVICES=$d python opencood/tools/track/eval_mot.py --model_dir "$dir/intermediate_epoch60"
