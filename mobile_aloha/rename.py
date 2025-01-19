# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import h5py

directory_path = os.path.join(ROOT, 'datasets/act/cup_in_plate')

old_group = '/observations/images/cam_left_wrist'
new_group = '/observations/images/left_wrist'
additional_group = '/observations/images/new_group'

def rename_group_in_hdf5_files(directory, old_group_name, new_group_name, add_group_name=None):
    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory path: {directory}")

    hdf5_files = [f for f in os.listdir(directory) if f.endswith(('.h5', '.hdf5'))]

    if not hdf5_files:
        print("No HDF5 files found in the directory.")

        return

    for filename in hdf5_files:
        file_path = os.path.join(directory, filename)

        try:
            with h5py.File(file_path, 'r+') as f:
                if old_group_name in f:
                    if new_group_name in f:
                        print(f"[SKIP] Group '{new_group_name}' already exists in file '{filename}'.")

                        continue

                    # 复制旧组到新组并删除旧组
                    f.copy(old_group_name, new_group_name)
                    del f[old_group_name]

                    print(f"[SUCCESS] Group '{old_group_name}' renamed to '{new_group_name}' in file '{filename}'.")
                else:
                    print(f"[NOT FOUND] Group '{old_group_name}' not found in file '{filename}'.")

                # 检查并添加新组
                if add_group_name not in f and add_group_name:
                    f.create_group(add_group_name)
                    f[add_group_name].create_dataset('example_dataset', data=[])  # Initialize with empty data

                    print(f"[SUCCESS] New group '{add_group_name}' added in file '{filename}'.")
        except Exception as e:
            print(f"[ERROR] An error occurred while processing file '{filename}': {e}")

rename_group_in_hdf5_files(directory_path, old_group, new_group, additional_group)