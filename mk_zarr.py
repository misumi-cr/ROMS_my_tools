#!/usr/bin/env python3

import xarray as xr
import dask
import glob

def process_file(file, variables_to_merge):
    ds = xr.open_dataset(file, chunks={'ocean_time': -1})
    # 指定された変数のみを選択
    return ds[variables_to_merge]

case_name='obtn_h040_s05.135'
variables_to_merge = ['temp', 'salt']
src_dir=f'/data44/misumi/roms_out/{case_name}/out'
dst_dir=f'/data44/misumi/roms_zarr_test'

# ファイルリストを取得
files = sorted(glob.glob(f'{src_dir}/{case_name}.a.00[1-5].nc'))

# 各ファイルに対して遅延処理を適用
lazy_datasets = [dask.delayed(process_file)(f, variables_to_merge) for f in files]

# 遅延オブジェクトを計算し、データセットのリストを取得
datasets = dask.compute(*lazy_datasets)

# xarray.mergeを使用してデータセットを結合
merged_ds = xr.merge(datasets, compat='equals', join='outer')

# 重複する時間を削除（必要な場合）
merged_ds = merged_ds.sel(time=~merged_ds.indexes['ocean_time'].duplicated(keep='first'))

# 結果をZarr形式で保存
merged_ds.chunk({'ocean_time': 1}).to_zarr(f'{dst_dir}/{case_name}')