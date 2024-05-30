gpu_id=$1


comm_thre=0
#baseline=('opencood/logs_HEAL/Dairv2x_attfuse_new' 'opencood/logs_HEAL/Dairv2x_DiscoNet' 'opencood/logs_HEAL/Dairv2x_where2comm' 'opencood/logs_HEAL/codebook_dairv2x_size16' 'opencood/logs_HEAL/codebook_dairv2x_size4')
baseline=('opencood/logs_HEAL/Dairv2x_DiscoNet')


#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir 'opencood/logs_HEAL/Dairv2x_V2VNet_new'  --delay_time 0 --model_name baseline --modal 0 --range "102.4,48" --note "noise" --result_name "lidar_only_noise"

for i in ${baseline[*]}
do
 echo baseline_$i
 CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir $i --modal 0 --delay_time 0 --model_name baseline --range "102.4,51.2" --note "noise" --result_name "lidar_only_noise"
done

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir 'opencood/logs_HEAL/Dairv2x_V2VNet_new'  --delay_time 0 --model_name baseline --modal 3 --range "102.4,48" --note "noise" --result_name "egocamera_otherlidar_noise"

for i in ${baseline[*]}
do
 echo baseline_$i
 CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir $i --modal 3 --delay_time 0 --model_name baseline --range "102.4,51.2" --note "noise" --result_name "egocamera_otherlidar_noise"
done

#CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir 'opencood/logs_HEAL/Dairv2x_V2VNet_new'  --delay_time 0 --model_name baseline --modal 1 --range "102.4,48" --note "noise" --result_name "camera_only_noise"

for i in ${baseline[*]}
do
 echo baseline_$i
 CUDA_VISIBLE_DEVICES=1 python opencood/tools/inference_comm_w_noise.py --model_dir $i --modal 1 --delay_time 0 --model_name baseline --range "102.4,51.2" --note "noise" --result_name "camera_only_noise"
done
