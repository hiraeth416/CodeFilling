gpu_id=$1


comm_thre=0
# baseline=('opencood/logs/Dairv2x_attfuse_new')
# baseline=('opencood/logs/Dairv2x_DiscoNet')
baseline=('opencood/logs_HEAL/Dairv2x_DiscoNet')


latency=(0 1 2 3 5)
#modal=(0)

#for j in ${latency[*]}
#do
#for i in ${baseline[*]}
#do
# echo latency_$j
# echo baseline_$i
# CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference_comm_bp_plus.py --model_dir $i --delay_time $j --modal 0 --result_name "lidar_only" --model_name baseline_latency --range "102.4,51.2"
#done
#done


for j in ${latency[*]}
do
for i in ${baseline[*]}
do
 echo latency_$j
 echo baseline_$i
 CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference_comm_bp_plus.py --model_dir $i --delay_time $j --modal 1 --result_name "camera_only" --model_name baseline_latency --range "102.4,51.2"
done
done
#
#
for j in ${latency[*]}
do
for i in ${baseline[*]}
do
 echo latency_$j
 echo baseline_$i
 CUDA_VISIBLE_DEVICES=7 python opencood/tools/inference_comm_bp_plus.py --model_dir $i --delay_time $j --modal 3 --result_name "egocamera_otherlidar" --model_name baseline_latency --range "102.4,51.2"
done
done