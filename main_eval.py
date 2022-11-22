from __future__ import print_function, division

import os
import numpy as np
from utils.data_utils import loading_scene_list

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import random
import json
from tqdm import tqdm

from utils.model_util import ScalarMeanTracker
from runners import a3c_val


def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    scenes = loading_scene_list(args)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = mp.Queue()
    args.learned_loss = False
    args.num_steps = 50
    target = a3c_val
    args.model_phase = 'test'  # add for whether compute kl  

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
                scenes[rank],
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    visualizations = []

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)
            visualizations.append(train_result['tools'])

        tracked_means = train_scalars.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
    if args.results_path == '.':
        pass
    else:
        if not os.path.exists(args.results_path):
            os.mkdir(args.results_path)
        
    with open(args.results_path + '/' + args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)

    # best results zsx
    print(args.record_best)
    if args.record_best:
        print('####')
        fb_name = args.results_path + '/' +  args.results_json.split('.json')[0] + '_best.json'
        if not os.path.exists(fb_name):
            with open(fb_name, "w") as fb:
                json.dump(tracked_means, fb, sort_keys=True, indent=4)
        elif not os.path.getsize(fb_name):
            with open(fb_name, "w") as fb:
                json.dump(tracked_means, fb, sort_keys=True, indent=4)
        else:
            with open(fb_name, "r") as fb:
                results_best = json.load(fb)
            if tracked_means["success"] > results_best["success"]:
                with open(fb_name, "w") as fb:
                    json.dump(tracked_means, fb, sort_keys=True, indent=4)
 
    # byb add
    # with open(args.results_json, "r") as f:
    #     results = json.load(f)
    # if results["success"] > 0.60:
    #     present_model_dict={'model': args.present_model,'rank':rank}
    #     with open('./sp_test/location_result_{}'.format(results["success"]), "w") as fp:
    #         json.dump(tracked_means, fp, sort_keys=True, indent=4)
    #         json.dump(present_model_dict,fp)
   
    visualization_dir = args.results_path + '/' + 'visualization_files'
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)

    with open(os.path.join(visualization_dir, 'visualization_' + args.results_json), 'w') as wf:
        json.dump(visualizations, wf)
