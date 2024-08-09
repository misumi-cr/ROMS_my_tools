#!/usr/bin/env python3


def select_interior(ds):
    """
    discard "exterior" u,v,rho-points to build a symetric grid
        Parameters:
            ds (xarray.Dataset): ROMS dataset
    """
    ds = ds.isel(xi_rho=slice(1,-1), eta_rho=slice(1,-1))
    if 'xi_v' in ds.dims:
        ds = ds.isel(xi_v=slice(1,-1))
    if 'eta_u' in ds.dims:
        ds = ds.isel(eta_u=slice(1,-1))
    return ds

import xarray as xr
import dask
import glob
import pandas as pd
import numpy as np
from xgcm import Grid

def process_file(file, variables_to_merge):
    ds = xr.open_dataset(file, chunks={'ocean_time': -1})
    return ds[variables_to_merge]

grid_name='/data44/misumi/obtn_zarr/obtn_mount_adcp-z5_grd-17cm_nearest_rx10.nc'
case_name='obtn_h040_s05.135'
variables_to_merge = ['temp', 'salt']
src_dir=f'/data44/misumi/roms_out/{case_name}/out'
dst_dir=f'/data44/misumi/roms_zarr_test'


# グリッドファイル取得と処理
ds_grid=xr.open_dataset(grid_name)
ds_grid=ds_grid.drop_vars(['hraw','lon_vert','lat_vert','x_vert','y_vert','spherical'])
ds_grid=select_interior(ds_grid)

# ファイルリストを取得
files = sorted(glob.glob(f'{src_dir}/{case_name}.a.00[1-5].nc'))

# 各ファイルに対して遅延処理を適用
lazy_datasets = [dask.delayed(process_file)(f, variables_to_merge) for f in files]

# 遅延オブジェクトを計算し、データセットのリストを取得
datasets = dask.compute(*lazy_datasets)

# xarray.concatを使用してデータセットを結合
concat_ds = xr.concat(datasets, dim='ocean_time')

concat_ds = select_interior(concat_ds)

# 重複する時間を削除（必要な場合）
unique_times = ~pd.Index(concat_ds.ocean_time.values).duplicated(keep='first')
concat_ds = concat_ds.isel(ocean_time=unique_times)

# 結果をZarr形式で保存
concat_ds.chunk({'ocean_time': 1}).to_zarr(f'{dst_dir}/{case_name}')