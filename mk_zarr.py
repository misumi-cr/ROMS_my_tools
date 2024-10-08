#!/usr/bin/env python3
import os
import shutil
import xarray as xr
import dask
import glob
import pandas as pd
import numpy as np
from xgcm import Grid
import datetime as dt
import numpy.ma as ma

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

def add_coords(ds):
    """ set coordinate variables as xarray coordinates
        Parameters:
            ds (xarray.Dataset): ROMS dataset
    """
#    ds = ds.set_coords(['Cs_r', 'Cs_w', 'hc', 'h', 'Vtransform', 'time',
#                        'lon_rho', 'lon_v', 'lon_u', 'lon_psi',
#                        'lat_rho', 'lat_v', 'lat_u', 'lat_psi'])
    ds = ds.set_coords(['Cs_r', 'Cs_w', 'hc', 'h',
                        'lon_rho', 'lon_v', 'lon_u', 'lon_psi',
                        'lat_rho', 'lat_v', 'lat_u', 'lat_psi'])
    return ds

def compute_depth_layers(ds, grid, hmin=-0.1):
    """ compute depths of ROMS vertical levels (Vtransform = 2) """

    # compute vertical transformation functional
    S_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    S_w = (ds.hc * ds.s_w + ds.Cs_w * ds.h) / (ds.hc + ds.h)
    
    # compute depth of rho (layers) and w (interfaces) points
    z_rho = ds.zeta + (ds.zeta + ds.h) * S_rho
    z_w = ds.zeta + (ds.zeta + ds.h) * S_w

    z_rho.data=ma.masked_outside(z_rho.data,-1.e4,1.e4)
    z_w.data=ma.masked_outside(z_w.data,-1.e4,1.e4)

    # transpose arrays and fill NaNs with a minimal depth
    ds['z_rho'] = z_rho.transpose(*('time', 's_rho','yh','xh'),
                                  transpose_coords=False).fillna(hmin)
    
    ds['z_w'] = z_w.transpose(*('time', 's_w','yh','xh'),
                                  transpose_coords=False).fillna(hmin)
    
    # interpolate depth of levels at U and V points
    ds['z_u'] = grid.interp(ds['z_rho'], 'X', boundary='fill')
    ds['z_v'] = grid.interp(ds['z_rho'], 'Y', boundary='fill')
    
    # compute layer thickness as difference between interfaces
    ds['dz'] = grid.diff(ds['z_w'], 'Z')
#    
#    # add z_rho and z_w to xarray coordinates
#    ds = ds.set_coords(['z_rho', 'z_w', 'z_v', 'z_u'])
    
    return ds

def set_time(ds):
    mdl_time=[]
    fig_time=[]
    for tm in ds["time"].data:
        mdl_time.append(dt.datetime(2000+tm.year,tm.month,tm.day,tm.hour,0,0))
        fig_time.append(dt.datetime(2000,tm.month,tm.day,tm.hour,0,0))
    ds["mdl_time"]=xr.DataArray(np.array(mdl_time),dims=["time"])
    ds["fig_time"]=xr.DataArray(np.array(fig_time),dims=["time"])
    ds=ds.set_index(time="mdl_time")
    ds=ds.set_coords(["fig_time"])
    return ds


if __name__ == "__main__":

    dask.config.set(**{'array.slicing.split_large_chunks': True})  # True: optimize chunk automatically

    grid_name='grd/Grd_slope500m_231020.nc'
    case_name='simple_004trc'

    variables=['temp','salt','dye_01']
    variables=variables+['zeta'] # required to calculate depth

    src_dir=f'/data45/misumi/roms_out/{case_name}'
    dst_dir=f'/data45/misumi/roms_zarr/{case_name}'

    # input files
    flist=sorted(glob.glob(f'{src_dir}/{case_name}.a.00[12].nc'))

    # ディレクトリが存在する場合、ユーザーに確認を求める
    if os.path.exists(dst_dir):
        while True:
            response = input(f"ディレクトリ '{dst_dir}' は既に存在します。削除しますか？ (y/n): ").lower()
            if response in ['y', 'n']:
                break
            print("'y' または 'n' で回答してください。")
    
        if response == 'y':
            print(f"ディレクトリ '{dst_dir}' を削除します。")
            shutil.rmtree(dst_dir)
        else:
            print("処理を中止します。")
            exit()
    
    # グリッドファイル取得と処理
    ds_grid=xr.open_dataset(grid_name)
    ds_grid=select_interior(ds_grid)
    drop_vars=list(ds_grid.variables) # grid variables were obtained from the grid file, making a list to eliminate from the data files
    
    ds0=[process_file(f,variables) for f in flist]
    
    # xarray.concatを使用してデータセットを結合
    ds0_concat=xr.concat(ds0, dim='ocean_time')
    ds0_concat=ds0_concat.drop_vars(drop_vars,errors='ignore') # ignore errors due to eliminating variables that are not listed on drop_vars

    ds0_concat=select_interior(ds0_concat)
    ds0_concat=xr.merge([ds0_concat,ds_grid])
    ds0_concat=rename_dims(ds0_concat)
    ds0_concat=add_coords(ds0_concat)
    ds0_concat=set_time(ds0_concat)

    ds0_concat=ds0_concat.chunk({'time': 1})
    grid=Grid(ds0_concat, coords={'X': {'center': 'xh', 'outer': 'xq'},
                                  'Y': {'center': 'yh', 'outer': 'yq'},
                                  'Z': {'center': 's_rho', 'outer': 's_w'}},
              periodic=False)

    ds0_concat=compute_depth_layers(ds0_concat,grid)

    # 結果をZarr形式で保存
    ds0_concat.chunk({'time': 1}).to_zarr(f'{dst_dir}')
