#!/usr/bin/env python

import argparse
import dask
import fsspec
import numpy as np
import os
import time
import xarray as xr
import zarr.storage

from numcodecs import Zstd   # , Blosc
from dask.distributed import Client


def main():
    parser = argparse.ArgumentParser(description='Create cloud-optimized zarr files from WRF CONUS404')
    parser.add_argument('-i', '--index', help='Index to process', type=int, required=True)
    parser.add_argument('-s', '--step', help='Number of indices to process from start index', type=int, required=True)
    parser.add_argument('-b', '--base_dir', help='Directory to work in', required=False, default=None)
    parser.add_argument('-d', '--dst_dir', help='Path to destination zarr store', required=True)
    parser.add_argument('-z', '--zarr_dir', help='Location of source zarr files', required=True)

    args = parser.parse_args()

    print(f'HOST: {os.environ.get("HOSTNAME")}')
    print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    target_pat = 'target_'
    first_idx = args.index
    idx_span = args.step
    last_idx = first_idx + idx_span

    base_dir = os.path.realpath(args.base_dir)
    outzarr_dir = os.path.realpath(args.dst_dir)

    src_zarr = os.path.realpath(f'{args.zarr_dir}/{target_pat}')

    dst_zarr = outzarr_dir

    print(f'{idx_span=}')
    print(f'Processing indexes: {first_idx} to {last_idx}')
    print('-'*60)
    print(f'{base_dir=}')
    print(f'{outzarr_dir=}')
    print(f'{src_zarr=}')
    print(f'{dst_zarr=}')

    time_cnk = 24

    fs = fsspec.filesystem('file')

    # Build dictionary of candidate target paths to process
    target_dict = {idx: f'{args.zarr_dir}/{target_pat}{idx:05d}' for idx in range(first_idx, last_idx)}

    print(f'Index start: {first_idx}; Index end: {last_idx - 1}')

    t1_proc = time.time()
    # Start up the cluster
    # client = Client(n_workers=8, threads_per_worker=1, memory_limit='24GB')
    with dask.config.set({"distributed.scheduler.worker-saturation": 1.0}):
        client = Client(n_workers=10, threads_per_worker=2, diagnostics_port=None)   # , memory_limit='24GB')

    # Change the default compressor to Zstd
    # NOTE: 2022-08: The LZ-related compressors seem to generate random errors
    #       when part of a job on denali or tallgrass.
    zarr.storage.default_compressor = Zstd(level=9)

    print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    # Max total memory in gigabytes for cluster
    total_mem = sum(vv['memory_limit'] for vv in client.scheduler_info()['workers'].values()) / 1024**3
    total_threads = sum(vv['nthreads'] for vv in client.scheduler_info()['workers'].values())
    print(f'Total memory: {total_mem:0.1f} GB')
    print(f'Number of threads: {total_threads}')

    print(f'Client startup time: {time.time() - t1_proc:0.3f} s', flush=True)

    t1_proc = time.time()
    # if first_idx == 0:
    #     # Open first and last zarr file to get date range
    #     ds0 = xr.open_dataset(zlist[0], engine='zarr', chunks={})
    #     ds1 = xr.open_dataset(zlist[-1], engine='zarr', chunks={})
    #
    #     # TODO: the freq argument must reflect the time interval (e.g hourly, daily)
    #     dates = pd.date_range(start=ds0.time[0].values, end=ds1.time[-1].values, freq='1d')
    #
    #     # Have to drop the constant variables (e.g. variables having no time dimension)
    #     drop_vars = ['LANDMASK', 'lat', 'lon', 'x', 'y', 'crs']
    #
    #     source_dataset = ds0.drop_vars(drop_vars, errors='ignore')
    #
    #     template = (source_dataset.chunk().pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(time=len(dates)))
    #     template['time'] = dates
    #     template = template.chunk({'time': time_cnk})
    #
    #     # Writes no data (yet)
    #     template.to_zarr(dst_zarr, compute=False, consolidated=True, mode='w')
    #
    #     # Writes the data
    #     ds0.drop_vars(drop_vars).to_zarr(dst_zarr, region={'time': slice(0, time_cnk)})
    #
    #     # Add the wrf constants
    #     add_vars = ['LANDMASK', 'lat', 'lon', 'x', 'y', 'crs']
    #     ds0[add_vars].to_zarr(dst_zarr, mode='a')
    #     print(f'  Index {first_idx} (pre-create output): {time.time() - t1_proc:0.3f} s')

    # Get the time values from the destination zarr store
    ds_dst = xr.open_dataset(dst_zarr, engine='zarr', mask_and_scale=True, chunks={})
    dst_time = ds_dst.time.values

    # for i in range(first_idx, last_idx):
    for idx, cfile in target_dict.items():
        t1 = time.time()
        start = idx * time_cnk
        stop = (idx + 1) * time_cnk

        try:
            dsi = xr.open_dataset(cfile, engine='zarr', mask_and_scale=True, chunks={})
        except FileNotFoundError:
            print(f'{cfile} does not exist; skipping.')
            continue

        st_date_src = dsi.time.values[0]
        en_date_src = dsi.time.values[-1]
        st_time_idx = np.where(dst_time == st_date_src)[0].item()
        en_time_idx = np.where(dst_time == en_date_src)[-1].item() + 1

        print(f'  time slice: {start}, {stop} = {stop-start}')
        print(f'  dst time slice: {st_time_idx}, {en_time_idx} = {en_time_idx - st_time_idx}')

        # print(zlist[i])
        # dsi = xr.open_dataset(zlist[i], engine='zarr', chunks={})
        dsi.to_zarr(dst_zarr, region={'time': slice(st_time_idx, en_time_idx)})
        print(f'  Index {idx}: {time.time() - t1:0.3f} s', flush=True)
        print('-'*30)

    client.close()
    if dask.config.get("temporary-directory") == '/dev/shm':
        try:
            fs.rm(f'/dev/shm/dask-worker-space', recursive=True)
        except FileNotFoundError:
            pass

    print(f'Total time: {(time.time() - t1_proc) / 60.0:0.3f} m')


if __name__ == '__main__':
    main()
