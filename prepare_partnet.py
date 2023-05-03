import os
import json
import argparse

import h5py
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=10000)
parser.add_argument('--num_ins', type=int, default=200)
parser.add_argument(
    '--merging_data_path', type=str,
    default='/projects/katefgroup/part_based/partnet_dataset/stats/after_merging_label_ids/'
)
parser.add_argument(
    '--in_path', type=str,
    default='/projects/katefgroup/datasets/partnet/ins_seg_h5/'
)
parser.add_argument(
    '--out_path', type=str,
    default='/projects/katefgroup/datasets/partnet/partnet_analogical/'
)
args = parser.parse_args()

NUM_INS = args.num_ins
print('Number of Instances: ', NUM_INS)
NUM_POINT = args.num_point
print('Number of Points: ', NUM_POINT)


def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        label = fin['label'][:]
        return pts, label


def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)


def save_h5(fn, pts, gt_label, gt_mask, gt_valid, gt_other_mask):
    fout = h5py.File(fn, 'w')
    fout.create_dataset(
        'pts', data=pts,
        compression='gzip', compression_opts=4, dtype='float32'
    )
    fout.create_dataset(
        'gt_label', data=gt_label,
        compression='gzip', compression_opts=4, dtype='uint8'
    )
    fout.create_dataset(
        'gt_mask', data=gt_mask,
        compression='gzip', compression_opts=4, dtype='bool'
    )
    fout.create_dataset(
        'gt_valid', data=gt_valid,
        compression='gzip', compression_opts=4, dtype='bool'
    )
    fout.create_dataset(
        'gt_other_mask', data=gt_other_mask,
        compression='gzip', compression_opts=4, dtype='bool'
    )
    fout.close()


def reformat_data(in_h5_fn, out_h5_fn, split, file_counter=0):
    # Load data: this loads multiple instances from an h5 file
    record = load_json(in_h5_fn.replace('.h5', '.json'))
    pts, label = load_h5(in_h5_fn)
    pts = pts[:, :NUM_POINT, :]  # keep the first NUM_POINT points
    label = label[:, :NUM_POINT]
    n_shape = label.shape[0]
    if n_shape == 0:
        print("Skipping as no sample exists in this h5 file")
        return None

    # Store every instance to an array
    gt_label = np.zeros((n_shape, NUM_POINT), dtype=np.uint8)
    gt_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=bool)
    gt_valid = np.zeros((n_shape, NUM_INS), dtype=bool)
    gt_other_mask = np.zeros((n_shape, NUM_POINT), dtype=bool)
    bad_instance_idx = []  # some idxs will be dropped
    for i in (range(n_shape)):
        cur_label = label[i, :NUM_POINT]
        cur_record = record[i]
        cur_tot = 0
        for item in cur_record['ins_seg']:
            if item['part_name'] in part_name_list_level:
                selected = np.isin(cur_label, item['leaf_id_list'])
                gt_label[i, selected] = part_name_list[item['part_name']]
                gt_mask[i, cur_tot, selected] = True
                gt_valid[i, cur_tot] = True
                cur_tot += 1
        # Filtering conditions
        is_bad = (
            gt_valid[i].astype(int).sum() == 0
            or 0 in gt_mask[i, gt_valid[i]].astype(int).sum(1)
        )
        if is_bad:
            bad_instance_idx.append(i)
        gt_other_mask[i, :] = (gt_label[i, :] == 0)

    # Drop bad instances
    gt_label = np.delete(gt_label, bad_instance_idx, 0)
    gt_mask = np.delete(gt_mask, bad_instance_idx, 0)
    gt_valid = np.delete(gt_valid, bad_instance_idx, 0)
    gt_other_mask = np.delete(gt_other_mask, bad_instance_idx, 0)
    pts = np.delete(pts, bad_instance_idx, 0)
    n_shape = gt_label.shape[0]

    # Store each instance as a separate file
    for i in (range(n_shape)):
        pts_new = pts[[i]]
        gt_label_new = gt_label[[i]]
        gt_mask_new = gt_mask[[i]]
        gt_valid_new = gt_valid[[i]]
        gt_other_mask_new = gt_other_mask[[i]]
        out_h5_fn_i = os.path.join(out_h5_fn, split + f"{file_counter:05d}.h5")
        file_counter += 1
        save_h5(
            out_h5_fn_i,
            pts_new, gt_label_new, gt_mask_new, gt_valid_new, gt_other_mask_new
        )
    return file_counter


# main
partnet_categories = [
    "Bag", "Bed", "Bottle", "Bowl", "Chair",
    "Clock", "Dishwasher", "Display", "Door", "Earphone",
    "Faucet", "Hat", "Keyboard", "Knife", "Lamp",
    "Laptop", "Microwave", "Mug", "Refrigerator", "Scissors",
    "StorageFurniture", "Table", "TrashCan", "Vase"
]
levels = [1, 2, 3]
splits = ["train", "test"]

for category in partnet_categories:
    for level_id in levels:
        for split in splits:

            # load meta data files level-wise
            stat_in_fn_level = os.path.join(
                args.merging_data_path,
                '%s-level-%d.txt' % (category, level_id)
            )
            # load meta data files
            stat_in_fn = os.path.join(
                args.merging_data_path,
                '%s.txt' % (category)
            )
            part_name_list = {}
            with open(stat_in_fn, 'r') as fin:
                for item in fin.readlines():
                    part_name = item.rstrip().split()[1]
                    part_idx = int(item.rstrip().split()[0])
                    part_name_list[part_name] = part_idx
            if not os.path.isfile(stat_in_fn_level):
                print("Doesn't exist ...", category, level_id, split)
                continue
            with open(stat_in_fn_level, 'r') as fin:
                part_name_list_level = [
                    item.rstrip().split()[1] for item in fin.readlines()
                ]

            # Processing and cleaning
            data_in_dir = os.path.join(args.in_path, category)
            data_out_dir = os.path.join(
                args.out_path,
                '%s-%d-orig' % (category, level_id)
            )
            os.makedirs(data_out_dir, exist_ok=True)
            print("PROCESSING ...", category, level_id, split)
            h5_fn_list = [
                item for item in os.listdir(data_in_dir)
                if item.endswith('.h5') and item.startswith('%s-' % split)
            ]
            file_counter = 0
            for item in tqdm(h5_fn_list):
                file_counter = reformat_data(
                    os.path.join(data_in_dir, item),
                    data_out_dir,
                    split, file_counter
                )
