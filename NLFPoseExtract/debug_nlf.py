smpl_path = "/workspace/ywh_data/DataProcessNew/dongman/smpl/ca8637a1f14435234a9f1f9bc31ca4bb.pkl"

import pickle

with open(smpl_path, 'rb') as f:
    smpl_data = pickle.load(f)
    data = smpl_data['pose']['joints3d_nonparam']

    breakpoint()
    print(data)