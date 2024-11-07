#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import dask
import datetime
import fsspec
import pandas as pd
import time

import xarray as xr
import zarr
import zarr.storage

from numcodecs import Zstd   # , Blosc
from dask.distributed import Client

from ..conus404_helpers import (set_file_path, set_target_path, read_metadata, get_maxmem_per_thread, rechunker_wrapper,
                                apply_metadata, build_hourly_filelist, delete_dir)

import ctypes


def trim_memory() -> int:
    libc = ctypes.CDLL('libc.so.6')
    return libc.malloc_trim(0)


def set_blosc():
    import numcodecs
    numcodecs.blosc.use_threads = False

    return numcodecs.blosc.use_threads


def main():
    parser = argparse.ArgumentParser(description='Create cloud-optimized zarr files from CONUS404 model output files')
    parser.add_argument('-i', '--index', help='Index to process', required=True)
    parser.add_argument('-b', '--base_dir', help='Directory to work in', required=False, default=None)
    parser.add_argument('-w', '--wrf_dir', help='Base directory for WRF model output files', required=True)
    parser.add_argument('-c', '--constants_file', help='Path to WRF constants', required=False, default=None)
    parser.add_argument('-v', '--vars_file', help='File containing list of variables to include in output',
                        required=True)
    parser.add_argument('-d', '--dst_dir', help='Location to store rechunked zarr files', required=True)
    parser.add_argument('-m', '--metadata_file', help='File containing metadata to include in zarr files',
                        required=True)

    args = parser.parse_args()

    temp_store = os.environ.get("RAM_SCRATCH")

    print(f'HOST: {os.environ.get("HOSTNAME")}')
    print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')
    print(f'RAM_SCRATCH: {temp_store}')

    if temp_store is None:
        # Try to use share memory even if the job doesn't set a directory
        temp_store = '/dev/shm/tmp'

    base_dir = os.path.realpath(args.base_dir)
    wrf_dir = os.path.realpath(args.wrf_dir)
    const_file = set_file_path(args.constants_file, base_dir)
    metadata_file = set_file_path(args.metadata_file, base_dir)
    proc_vars_file = set_file_path(args.vars_file, base_dir)

    # The scratch filesystem seems to randomly return false for pre-existing
    # directories when a lot of processes are using the same path. Try random
    # sleep to minimize this.
    try:
        target_store = f'{set_target_path(args.dst_dir, base_dir)}/target'
    except FileNotFoundError:
        print(f'{args.dst_dir} not found; trying again')
        time.sleep(10)
        target_store = f'{set_target_path(args.dst_dir, base_dir)}/target'

    print(f'{base_dir=}')
    print(f'{wrf_dir=}')
    print(f'{const_file=}')
    print(f'{metadata_file=}')
    print(f'{proc_vars_file=}')
    print(f'{target_store=}')
    print(f'{temp_store=}')
    print('-'*60)

    base_date = datetime.datetime(1979, 10, 1)
    num_days = 6
    delta = datetime.timedelta(days=num_days)

    # Start date is selected based on chunk index
    index_start = int(args.index)
    st_date = base_date + datetime.timedelta(days=num_days * index_start)
    en_date = st_date + delta - datetime.timedelta(days=1)

    print(f'{index_start=}')
    print(f'base_date: {base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'st_date: {st_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'en_date: {en_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{num_days=}')
    print(f'{delta=}')
    print('-'*60)

    if (st_date - base_date).days % num_days != 0:
        print(f'Start date must begin at the start of a {num_days}-day chunk')

    time_chunk = num_days * 24
    x_chunk = 175
    y_chunk = 175

    # Attributes that should be removed from all variables
    remove_attrs = ['FieldType', 'MemoryOrder', 'stagger', 'cell_methods']

    rename_dims = {'south_north': 'y', 'west_east': 'x',
                   'south_north_stag': 'y_stag', 'west_east_stag': 'x_stag',
                   'Time': 'time'}

    rename_vars = {'XLAT': 'lat', 'XLAT_U': 'lat_u', 'XLAT_V': 'lat_v',
                   'XLONG': 'lon', 'XLONG_U': 'lon_u', 'XLONG_V': 'lon_v'}

    # Read the metadata file for modifications to variable attributes
    var_metadata = read_metadata(metadata_file)

    # Start up the cluster
    dask.config.set({'distributed.scheduler.active-memory-manager.measure': 'optimistic'})
    dask.config.set({'distributed.worker.memory.recent-to-old-time': 60})
    dask.config.set({'distributed.worker.memory.rebalance.measure': 'managed'})
    dask.config.set({'distributed.worker.memory.spill': False})
    dask.config.set({'distributed.worker.memory.pause': False})
    dask.config.set({'distributed.worker.memory.terminate': False})
    dask.config.set({'distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_': 1})
    # dask.config.set({'distributed.logging.distributed': 'warning'})
    # dask.config.set({'distributed.logging.distributed.client': 'warning'})
    # dask.config.set({'distributed.logging.distributed.worker': 'warning'})
    # dask.config.set({'distributed.logging.bokeh': 'critical'})
    # dask.config.set({'distributed.logging.tornado': 'critical'})
    # dask.config.set({'distributed.logging.tornado__application': 'error'})
    # dask.config.set({'distributed.scheduler.worker-ttl': None})

    client = Client(n_workers=6, threads_per_worker=2)  # , diagnostics_port=None)
    client.amm.start()

    print(f'Number of workers: {len(client.ncores())}')
    print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    # Get the maximum memory per thread to use for chunking
    max_mem = get_maxmem_per_thread(client, max_percent=0.6, verbose=True)

    # Read variables to process
    df = pd.read_csv(proc_vars_file)
    print(f'Number of variables to process: {len(df)}')

    fs = fsspec.filesystem('file')

    start = time.time()

    cnk_idx = index_start
    c_start = st_date

    # Change the default compressor to Zstd
    # NOTE: 2022-08: The LZ-related compressors seem to generate random errors
    #       when part of a job on denali or tallgrass.
    zarr.storage.default_compressor = Zstd(level=9)
    # zarr.storage.default_compressor = Blosc(cname='blosclz', clevel=4, shuffle=Blosc.SHUFFLE)

    # Filename pattern for wrf2d hourly files
    file_pat = '{wrf_dir}/{wy_dir}/wrf2d_d01_{fdate.strftime("%Y-%m-%d_%H:%M:%S")}'

    while c_start < en_date:
        tstore_dir = f'{target_store}_{cnk_idx:05d}'

        # =============================================
        # Do some work here
        var_list = df['variable'].to_list()
        var_list.append('time')

        # Rechunker requires empty temp and target dirs
        delete_dir(fs, temp_store)
        delete_dir(fs, tstore_dir)
        time.sleep(3)  # Wait for files to be removed (necessary? hack!)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open netcdf source files
        t1 = time.time()

        try:
            job_files = build_hourly_filelist(num_days, c_start, wrf_dir, file_pat, verify=False)
            ds2d = xr.open_mfdataset(job_files, concat_dim='Time', combine='nested',
                                     parallel=True, coords="minimal", data_vars="minimal",
                                     engine='netcdf4', compat='override', chunks={})
        except FileNotFoundError:
            # Re-run the filelist build with the expensive verify
            job_files = build_hourly_filelist(num_days, c_start, wrf_dir, file_pat, verify=True)
            print(job_files[0])
            print(job_files[-1])
            print(f'Number of valid files: {len(job_files)}')

            ds2d = xr.open_mfdataset(job_files, concat_dim='Time', combine='nested',
                                     parallel=True, coords="minimal", data_vars="minimal",
                                     engine='netcdf4', compat='override', chunks={})

        if cnk_idx == 0:
            # Add the wrf constants during the first time chunk
            df_const = xr.open_dataset(const_file, decode_coords=True, chunks={})
            ds2d = ds2d.merge(df_const)

            for vv in df_const.variables:
                if vv in rename_vars:
                    var_list.append(rename_vars[vv])
                elif vv in rename_dims:
                    var_list.append(rename_dims[vv])
                else:
                    var_list.append(vv)
            df_const.close()

            ds2d = apply_metadata(ds2d, rename_dims, rename_vars, remove_attrs, var_metadata)
        else:
            # The rename_vars variable is only needed for the first rechunk index
            # when the constants file is added.
            ds2d = apply_metadata(ds2d, rename_dims, {}, remove_attrs, var_metadata)

        print(f'    Open mfdataset: {time.time() - t1:0.3f} s', flush=True)

        # with performance_report(filename=f'dask_perf_{args.index}.html'):
        rechunker_wrapper(ds2d[var_list], target_store=tstore_dir, temp_store=temp_store,
                          mem=max_mem, consolidated=True, verbose=False,
                          chunks={'time': time_chunk,
                                  'y': y_chunk, 'x': x_chunk,
                                  'y_stag': y_chunk, 'x_stag': x_chunk})

        end = time.time()
        print(f'Chunk: {cnk_idx}, elapsed time: {(end - start) / 60.:0.3f} minutes, {job_files[0]}')

        cnk_idx += 1
        c_start += delta

        client.run(trim_memory)

    client.close()
    print('-- rechunk done', flush=True)

    # Clear out the temporary storage
    delete_dir(fs, temp_store)

    if dask.config.get("temporary-directory") == '/dev/shm':
        delete_dir(fs, '/dev/shm/dask-worker-space')


if __name__ == '__main__':
    main()
