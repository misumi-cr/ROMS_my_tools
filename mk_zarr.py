#!/usr/bin/env python3

import xarray as xr
import dask
import glob

def process_file(file):
    ds=xr.open_dataset(file, chunks={'ocean_time': -1})
    return ds


src_dir='/data44/misumi/roms_out/obtn_h040_s05.135/out'
dst_dir='/data44/misumi/roms_zarr_test'

# ファイルリストを取得
files = sorted(glob.glob(f'{src_dir}/obtn_h040_s05.135.a.00[1-5].nc'))

print(files)

## 各ファイルに対して遅延処理を適用
#lazy_datasets = [dask.delayed(process_file)(f) for f in files]
#
## 遅延オブジェクトを計算し、データセットのリストを取得
#datasets = dask.compute(*lazy_datasets)
#
## xarray.mergeを使用してデータセットを結合
#merged_ds = xr.merge(datasets, compat='equals', join='outer')
#
## 重複する時間を削除（必要な場合）
#merged_ds = merged_ds.sel(time=~merged_ds.indexes['time'].duplicated(keep='first'))
#
## 結果をZarr形式で保存
#merged_ds.chunk({'time': 100}).to_zarr('output.zarr')