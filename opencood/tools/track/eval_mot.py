import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--model_dir", type=str, help='evaluated model')
    parser.add_argument("--model_name", type=str, help='model name')
    parser.add_argument("--note", type=str, help='note')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    log_root_dir = '/dssg/home/acct-umjpyb/umjpyb/junhaoge/OpenCOODv2/'
    args = parse_args()
    model_dir = args.model_dir
    model_name = args.model_name
    track_note = args.note

    print(args)

    result_dict = []
    
    # run eval for MOTA and MOTP
    base_cmd = 'python /dssg/home/acct-umjpyb/umjpyb/junhaoge/OpenCOODv2/opencood/tools/track/run_mot_challenge.py '
    base_cmd += '--BENCHMARK OPV2V --SPLIT_TO_EVAL test --DO_PREPROC False '
    base_cmd += f' --TRACKERS_TO_EVAL {log_root_dir}{model_dir}/track'
    base_cmd += f' --GT_FOLDER {log_root_dir}{model_dir}/track/gt/ '
    base_cmd += f' --TRACKERS_FOLDER {log_root_dir}{model_dir}/track '
    
    cmd = base_cmd + ' --METRICS CLEAR '
    
    os.system(cmd)

    # collect results
    eval_output_path = f'{log_root_dir}{model_dir}/track/pedestrian_summary.txt'
    eval_output_file = open(eval_output_path, 'r')
    # skip header
    eval_output_file.readline()
    perfs = eval_output_file.readline().split(' ')
    
    # MOTA and MOTP
    result_dict.append(float(perfs[0]))
    result_dict.append(float(perfs[1]))


    # run eval for other metrics
    cmd = base_cmd + ' --METRICS HOTA '
    os.system(cmd)
    
    # collect results
    eval_output_path = f'{log_root_dir}{model_dir}/track/pedestrian_summary.txt'
    eval_output_file = open(eval_output_path, 'r')
    # skip header
    eval_output_file.readline()
    perfs = eval_output_file.readline().split(' ')
    
    # HOTA DetA AssA DetRe DetPr AssRe AssPr LocA
    for ii in range(8):
        result_dict.append(float(perfs[ii]))
    
    df = pd.DataFrame([result_dict], columns=['MOTA', 'MOTP', 'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA'])
    df.to_csv(f'{log_root_dir}{model_dir}/track/logs.csv', sep=',', index=False)

    split_model_dir = '/'.join(model_dir.split('/')[:3])
    if track_note == "latency":
        comm_thre = str(int(model_dir.split('/')[3].split('_')[-1]) * 100)
    else:
        comm_thre = model_dir.split('/')[3].split('_')[-1]
    # comm_thre = model_dir.split('/')[3].split('_')[-2]
    tail = model_dir.split('/')[3].split('_')[-2]
    track_path = f'{log_root_dir}{split_model_dir}/{track_note}'
    if not os.path.exists(track_path):
        os.makedirs(track_path)
    collect_dir = f'{log_root_dir}{split_model_dir}/{track_note}/{model_name}_{track_note}.txt'
    with open(collect_dir, 'a') as f:
        f.write(comm_thre + ' ')
        f.write(' '.join([str(x) for x in result_dict]))
        f.write('\n')