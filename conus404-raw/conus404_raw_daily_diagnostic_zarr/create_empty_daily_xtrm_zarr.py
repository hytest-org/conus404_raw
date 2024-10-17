#!/usr/bin/env python

import argparse
import dask
import datetime
import os
import pandas as pd
import time
import xarray as xr
import zarr
import zarr.storage

from rich.console import Console
from rich import pretty
from numcodecs import Zstd   # , Blosc
from dask.distributed import Client, LocalCluster

from ..conus404_helpers import get_accum_types

pretty.install()
con = Console(record=True)


def main():
    parser = argparse.ArgumentParser(description='Create cloud-optimized daily diagnostic zarr store')
    parser.add_argument('-d', '--dst_zarr', help='Path of destination zarr store', required=True)
    parser.add_argument('-s', '--src_zarr', help='Path of source zarr store', required=True)
    parser.add_argument('--daterange',
                        help='Starting and ending calendar date (YYYY-MM-DD YYYY-MM-DD)',
                        nargs=2, metavar=('startDate', 'endDate'), required=True)
    parser.add_argument('--freq', help='Frequency to use for timesteps (e.g. "1h")', default='1h')

    args = parser.parse_args()

    con.print(f'HOST: {os.environ.get("HOSTNAME")}')
    con.print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    st_date = args.daterange[0]
    en_date = args.daterange[1]

    src_zarr = args.src_zarr

    # Output zarr store
    dst_zarr = args.dst_zarr

    time_chunk = 24
    x_chunk = 350
    y_chunk = 350

    con.print(f'Start date: {datetime.datetime.fromisoformat(st_date)}')
    con.print(f'End date: {datetime.datetime.fromisoformat(en_date)}')
    con.print(f'Time interval: {args.freq}')
    con.print(f'Source dataset: {src_zarr}')
    con.print(f'New dataset: {dst_zarr}')
    con.print(f'Chunks: {dict(time=time_chunk, y=y_chunk, x=x_chunk)}')
    con.print('-'*40)

    # daily_chunks = dict(y=y_chunk, x=x_chunk, y_stag=y_chunk, x_stag=x_chunk)
    dst_chunks = dict(y=y_chunk, x=x_chunk)

    con.print(f'dask tmp directory: {dask.config.get("temporary-directory")}')

    start_time = time.time()

    con.print('=== Open client ===')
    cluster = LocalCluster(n_workers=15, threads_per_worker=2, processes=True)

    with Client(cluster) as client:
        total_mem = sum(vv['memory_limit'] for vv in client.scheduler_info()['workers'].values()) / 1024**3
        total_threads = sum(vv['nthreads'] for vv in client.scheduler_info()['workers'].values())
        print(f'    --- Total memory: {total_mem:0.1f} GB; Threads: {total_threads}')

        # Change the default compressor to Zstd
        # NOTE: 2022-08: The LZ-related compressors seem to generate random errors
        #       when part of a job on denali or tallgrass.
        zarr.storage.default_compressor = Zstd(level=9)

        con.print('--- Create daily diagnostic zarr store ---')
        ds = xr.open_dataset(src_zarr, engine='zarr',
                             backend_kwargs=dict(consolidated=True),
                             decode_coords=False, chunks={})

        # Get integration information
        accum_types = get_accum_types(ds)
        drop_vars = accum_types.setdefault('constant', [])

        # Get the full date range from the hourly zarr store
        dates = pd.date_range(start=st_date, end=en_date, freq=args.freq)
        con.print(f'    number of timesteps: {len(dates)}')

        # Get all variables but the constant variables
        source_dataset = ds.drop_vars(drop_vars, errors='ignore')

        print('    --- Create template', end=' ')
        template = (source_dataset.chunk(dst_chunks).pipe(xr.zeros_like).isel(time=0, drop=True).expand_dims(time=len(dates)))
        template['time'] = dates
        template = template.chunk({'time': time_chunk})
        print(f'       {time.time() - start_time:0.3f} s', flush=True)

        # progress = Progress(
        #     TextColumn("[progress.description]{task.description}"),
        #     BarColumn(),
        #     TaskProgressColumn(),
        #     TimeRemainingColumn(),
        # )

        print('    --- Write template', flush=True, end=' ')
        # Writes no data (yet)
        template.to_zarr(dst_zarr, compute=False, consolidated=True, mode='w')
        print(f'       {time.time() - start_time:0.3f} s', flush=True)

        # Remove the existing chunk encoding for constant variables
        for vv in drop_vars:
            try:
                del ds[vv].encoding['chunks']
            except KeyError:
                pass

        # Add the wrf constants
        if len(drop_vars) > 0:
            print('    --- Write constant variables', end=' ')
            ds[drop_vars].chunk(dst_chunks).to_zarr(dst_zarr, mode='a')
            print(f'       {time.time() - start_time:0.3f} s', flush=True)

    con.print(f'Runtime: {(time.time() - start_time) / 60.:0.3f} m')


if __name__ == '__main__':
    main()
