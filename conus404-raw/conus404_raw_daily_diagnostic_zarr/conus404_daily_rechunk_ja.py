#!/usr/bin/env python

import os
import argparse
import dask
import datetime
import fsspec
import pandas as pd
import time
import xarray as xr
import zarr

from numcodecs import Zstd   # , Blosc
from dask.distributed import Client

from ..conus404_helpers import (set_file_path, set_target_path, read_metadata_xtrm, get_maxmem_per_thread,
                                delete_dir, apply_metadata_xtrm, build_daily_filelist, rechunker_wrapper)


def main():
    parser = argparse.ArgumentParser(description='Create cloud-optimized zarr files from WRF CONUS404 daily model output files')
    parser.add_argument('-i', '--index', help='Index to process', required=True)
    parser.add_argument('-b', '--base_dir', help='Directory to work in', required=False, default=None)
    parser.add_argument('-w', '--wrf_dir', help='Base directory for WRF model output files', required=True)
    parser.add_argument('-c', '--constants_file', help='Path to WRF constants', required=False, default=None)
    parser.add_argument('-v', '--vars_file', help='File containing list of variables to include in output', required=True)
    parser.add_argument('-d', '--dst_dir', help='Location to store rechunked zarr files', required=True)
    parser.add_argument('-m', '--metadata_file', help='File containing metadata to include in zarr files', required=True)

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
    print('-'*60)

    base_date = datetime.datetime(1979, 10, 1)
    num_days = 24
    delta = datetime.timedelta(days=num_days)

    # We specify a chunk index and the start date is selected based on that
    index_start = int(args.index)
    st_date = base_date + datetime.timedelta(days=num_days * index_start)
    en_date = st_date + delta - datetime.timedelta(days=1)

    # .time.dt.strftime("%Y-%m-%d %H:%M:%S")[0].values
    print(f'{index_start=}')
    print(f'base_date: {base_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'st_date: {st_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'en_date: {en_date.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{num_days=}')
    print(f'{delta=}')
    print('-'*60)

    if (st_date - base_date).days % num_days != 0:
        print(f'Start date must begin at the start of a {num_days}-day chunk')

    time_chunk = num_days
    x_chunk = 350
    y_chunk = 350

    # Variables to drop from the constants file
    drop_vars = ['BF', 'BH', 'C1F', 'C1H', 'C2F', 'C2H', 'C3F', 'C3H', 'C4F', 'C4H',
                 'CF1', 'CF2', 'CF3', 'CFN', 'CFN1', 'CLAT', 'COSALPHA', 'DN', 'DNW',
                 'DZS', 'E', 'F', 'FNM', 'FNP', 'HGT', 'ISLTYP', 'IVGTYP', 'LAKEMASK',
                 'LU_INDEX', 'MAPFAC_M', 'MAPFAC_MX', 'MAPFAC_MY',
                 'MAPFAC_U', 'MAPFAC_UX', 'MAPFAC_UY', 'MAPFAC_V', 'MAPFAC_VX', 'MAPFAC_VY',
                 'MAX_MSTFX', 'MAX_MSTFY', 'MF_VX_INV', 'MUB', 'P00', 'PB', 'PHB',
                 'P_STRAT', 'P_TOP', 'RDN', 'RDNW', 'RDX', 'RDY', 'SHDMAX', 'SHDMIN',
                 'SINALPHA', 'SNOALB', 'T00', 'TISO', 'TLP', 'TLP_STRAT', 'VAR',
                 'VAR_SSO', 'XLAND', 'XLAT_U', 'XLAT_V', 'XLONG_U', 'XLONG_V',
                 'ZETATOP', 'ZNU', 'ZNW', 'ZS']

    # Attributes that should be removed from all variables
    remove_attrs = ['FieldType', 'MemoryOrder', 'stagger', 'cell_methods']

    rename_dims = {'south_north': 'y', 'west_east': 'x', 'Time': 'time'}

    rename_vars = {'XLAT': 'lat', 'XLONG': 'lon',
                   'south_north': 'y', 'west_east': 'x'}

    # Read the metadata file for modifications to variable attributes
    var_metadata = read_metadata_xtrm(metadata_file)

    # Start up the cluster
    client = Client(n_workers=6, threads_per_worker=2)   # , memory_limit='24GB')

    print(f'Number of workers: {len(client.ncores())}')
    print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    # Get the maximum memory per thread to use for chunking
    max_mem = get_maxmem_per_thread(client, max_percent=0.7, verbose=False)

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

    # Filename pattern for wrf xtrm files
    # wrfxtrm_d01_2022-10-01_00:00:00
    file_pat = '{wrf_dir}/{wy_dir}/wrfxtrm_d01_{cdate.strftime("%Y-%m-%d_%H:%M:%S")}'

    while c_start < en_date:
        tstore_dir = f'{target_store}_{cnk_idx:05d}'

        try:
            job_files = build_daily_filelist(num_days, c_start, wrf_dir, file_pat, verify=False)
            ds2d = xr.open_mfdataset(job_files, concat_dim='Time', combine='nested',
                                     parallel=True, coords="minimal", data_vars="minimal",
                                     engine='netcdf4', compat='override', chunks={})
        except FileNotFoundError:
            # Re-run the filelist build with the expensive verify
            job_files = build_daily_filelist(num_days, c_start, wrf_dir, file_pat, verify=True)
            print(job_files[0])
            print(job_files[-1])
            print(f'Number of valid files: {len(job_files)}')

            ds2d = xr.open_mfdataset(job_files, concat_dim='Time', combine='nested',
                                     parallel=True, coords="minimal", data_vars="minimal",
                                     engine='netcdf4', compat='override', chunks={})

        # job_files = ch.build_daily_filelist(num_days, c_start, wrf_dir, file_pattern=file_pat)

        if len(job_files) < num_days:
            print(f'Number of files not equal to time chunk; adjusting to {len(job_files)}')
            num_days = len(job_files)
            time_chunk = num_days

        # =============================================
        # Do some work here
        var_list = df['variable'].to_list()
        var_list.append('time')

        # Rechunker requires empty temp and target dirs
        ch.delete_dir(fs, temp_store)
        ch.delete_dir(fs, tstore_dir)
        time.sleep(3)  # Wait for files to be removed (necessary? hack!)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open netcdf source files
        t1 = time.time()
        ds2d = xr.open_mfdataset(job_files, concat_dim='Time', combine='nested',
                                 parallel=True, coords="minimal", data_vars="minimal",
                                 compat='override', chunks={})

        if cnk_idx == 0:
            # Add the wrf constants during the first time chunk
            df_const = xr.open_dataset(const_file, decode_coords=False, chunks={})
            ds2d = ds2d.merge(df_const)

            for vv in df_const.variables:
                if vv in rename_vars:
                    var_list.append(rename_vars[vv])
                elif vv in rename_dims:
                    var_list.append(rename_dims[vv])
                else:
                    var_list.append(vv)
            df_const.close()

            ds2d = apply_metadata_xtrm(ds2d, rename_dims, rename_vars, remove_attrs, var_metadata)
        else:
            # The rename_vars variable is only needed for the first rechunk index
            # when the constants file is added.
            ds2d = apply_metadata_xtrm(ds2d, rename_dims, {}, remove_attrs, var_metadata)

        print(f'    Open mfdataset: {time.time() - t1:0.3f} s')

        rechunker_wrapper(ds2d[var_list], target_store=tstore_dir, temp_store=temp_store,
                          mem=max_mem, consolidated=True, verbose=False,
                          chunks={'time': time_chunk,
                                  'y': y_chunk, 'x': x_chunk,
                                  'y_stag': y_chunk, 'x_stag': x_chunk})

        end = time.time()
        print(f'Chunk: {cnk_idx}, elapsed time: {(end - start) / 60.:0.3f}, {job_files[0]}')

        cnk_idx += 1
        c_start += delta

    client.close()

    print('-- rechunk done', flush=True)

    # Clear out the temporary storage
    delete_dir(fs, temp_store)

    if dask.config.get("temporary-directory") == '/dev/shm':
        delete_dir(fs, '/dev/shm/dask-worker-space')


if __name__ == '__main__':
    main()
