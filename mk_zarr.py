#!/usr/bin/env python3
import xarray as xr
import dask
import glob
import pandas as pd
import numpy as np
from xgcm import Grid

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

def rename_dims(ds):
    """ rename dimensions
        Parameters:
            ds (xarray.Dataset): ROMS dataset
    """
    ds = ds.rename({'xi_rho': 'xh', 'xi_v': 'xh', 'xi_u': 'xq', 'xi_psi': 'xq',
                    'eta_rho': 'yh', 'eta_v': 'yq', 'eta_u': 'yh', 'eta_psi': 'yq',
                    'ocean_time': 'time'
                    })
    return ds

def process_file(file, variables_to_merge):
    ds = xr.open_dataset(file, chunks={'ocean_time': -1})
    return ds[variables_to_merge]

def process_grid(file):
    ds = xr.open_dataset(file)
    return ds

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
files=sorted(glob.glob(f'{src_dir}/{case_name}.a.00[1-5].nc'))

ds0=[process_file(f, variables_to_merge) for f in files]

# xarray.concatを使用してデータセットを結合
ds0_concat=xr.concat(ds0, dim='ocean_time')
ds0_concat=select_interior(ds0_concat)
ds0_concat=xr.merge([ds0_concat,ds_grid])
ds0_concat=rename_dims(ds0_concat)

# 重複する時間を削除（必要な場合）
unique_times = ~pd.Index(ds0_concat.ocean_time.values).duplicated(keep='first')
ds0_concat = ds0_concat.isel(ocean_time=unique_times)

# 結果をZarr形式で保存
ds0_concat.chunk({'ocean_time': 1}).to_zarr(f'{dst_dir}/{case_name}')