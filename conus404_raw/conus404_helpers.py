import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import os
import pandas as pd
import rechunker
import time
import xarray as xr
import zarr


def get_accum_types(ds):
    """
    Returns dictionary of acummulation types
    """

    accum_types = {}

    for vv in ds.variables:
        try:
            # accum_types[vv] = ds[vv].attrs['integration_length']
            accum_types.setdefault(ds[vv].attrs['integration_length'], [])

            if vv in ['I_ACLWDNB', 'I_ACLWUPB', 'I_ACSWDNB', 'I_ACSWDNT', 'I_ACSWUPB']:
                accum_types.setdefault('accumulated since 1979-10-01 00:00:00 bucket', [])
                accum_types['accumulated since 1979-10-01 00:00:00 bucket'].append(vv)
            else:
                accum_types.setdefault(ds[vv].attrs['integration_length'], [])
                accum_types[ds[vv].attrs['integration_length']].append(vv)
        except KeyError:
            if 'time' in ds[vv].dims:
                accum_types.setdefault('instantaneous', [])
                accum_types['instantaneous'].append(vv)
            else:
                accum_types.setdefault('constant', [])
                accum_types['constant'].append(vv)

    return accum_types


def apply_metadata(ds, rename_dims, rename_vars, remove_attrs, var_metadata):
    avail_dims = ds.dims.keys()
    rename_dims_actual = {}

    # Only change dimensions that exist in dataset
    for kk, vv in rename_dims.items():
        if kk in avail_dims:
            rename_dims_actual[kk] = vv

    ds = ds.rename(rename_dims_actual)
    if len(rename_vars) > 0:
        ds = ds.rename_vars(rename_vars)
    ds = ds.assign_coords({'time': ds.XTIME})
    # 2024-01-22 PAN: Something changed and now XTIME still has to be dropped from the dataset, whereas before it
    #                 was renamed to 'time'
    ds = ds.drop_vars(['XTIME'])

    # Modify the attributes
    for cvar in ds.variables:
        # Remove unneeded attributes, update the coordinates attribute
        for cattr in list(ds[cvar].attrs.keys()):
            if cattr in remove_attrs:
                del ds[cvar].attrs[cattr]

        # Apply new/updated metadata
        if cvar in var_metadata.index:
            for kk, vv in var_metadata.loc[cvar].dropna().to_dict().items():
                if kk in ['coordinates']:
                    # Add/modify encoding
                    ds[cvar].encoding.update({kk: vv})
                else:
                    ds[cvar].attrs[kk] = vv

    return ds


def apply_metadata_xtrm(ds, rename_dims, rename_vars, remove_attrs, var_metadata):
    avail_dims = ds.dims.keys()
    rename_dims_actual = {}

    # Only change dimensions that exist in dataset
    for kk, vv in rename_dims.items():
        if kk != 'Time' and kk in avail_dims:
            rename_dims_actual[kk] = vv

    ds = ds.rename(rename_dims_actual)

    if len(rename_vars) > 0:
        ds = ds.rename_vars(rename_vars)

    # The daily wrfxtrm files lack the XTIME variable but does have
    # the Times variable with the time as a string; create the time
    # coordinate variable from this.

    t_str = ds['Times'].values
    new_times = [datetime.datetime.strptime(tt.tobytes().decode('ascii'), '%Y-%m-%d_%H:%M:%S') for tt in t_str]

    ds['time'] = ('Time', new_times)
    ds = ds.rename({'Time': 'time'})
    ds = ds.assign_coords({'time': ds.time})
    ds = ds.drop('Times')

    # Modify the attributes
    for cvar in ds.variables:
        # Remove unneeded attributes, update the coordinates attribute
        for cattr in list(ds[cvar].attrs.keys()):
            if cattr in remove_attrs:
                del ds[cvar].attrs[cattr]

        # Apply new/updated metadata
        if cvar in var_metadata.index:
            for kk, vv in var_metadata.loc[cvar].dropna().to_dict().items():
                if kk in ['coordinates']:
                    # Add/modify encoding
                    ds[cvar].encoding.update({kk: vv})
                else:
                    ds[cvar].attrs[kk] = vv

    return ds


def build_daily_filelist(num_days, c_start, wrf_dir, file_pattern, verify=False):
    """
    Build a list of file paths
    """

    job_files = []

    for dd in range(num_days):
        cdate = c_start + datetime.timedelta(days=dd)

        wy_dir = f'WY{cdate.year}'
        if cdate >= datetime.datetime(cdate.year, 10, 1):
            wy_dir = f'WY{cdate.year+1}'

        # 201610010000.LDASIN_DOMAIN1
        # file_pat = f'{wrf_dir}/{wy_dir}/{fdate.strftime("%Y%m%d%H%M")}.LDASIN_DOMAIN1'
        file_pat = eval(f"f'{file_pattern}'")

        if verify:
            # Verifying the existence of each file can put a heavy load on Lustre filesystems
            # Only call this function with verify turned on when it's needed (e.g. when open_mfdataset fails).
            if os.path.exists(file_pat):
                job_files.append(file_pat)
            else:
                if cdate.month == 10 and cdate.day == 1:
                    # Sometimes a dataset includes the first hour of the next water year in
                    # each water year directory; we need this file.
                    wy_dir = f'WY{cdate.year}'
                    file_pat = eval(f"f'{file_pattern}'")

                    if os.path.exists(file_pat):
                        job_files.append(file_pat)

                    break
        else:
            job_files.append(file_pat)
    return job_files


def build_hourly_filelist(num_days, c_start, wrf_dir, file_pattern, verify=False):
    """
    Build a list of file paths
    """

    job_files = []

    for dd in range(num_days):
        cdate = c_start + datetime.timedelta(days=dd)

        wy_dir = f'WY{cdate.year}'
        if cdate >= datetime.datetime(cdate.year, 10, 1):
            wy_dir = f'WY{cdate.year+1}'

        for hh in range(24):
            fdate = cdate + datetime.timedelta(hours=hh)

            # 201610010000.LDASIN_DOMAIN1
            # file_pat = f'{wrf_dir}/{wy_dir}/{fdate.strftime("%Y%m%d%H%M")}.LDASIN_DOMAIN1'
            file_pat = eval(f"f'{file_pattern}'")

            if verify:
                # Verifying the existence of each file can put a heavy load on Lustre filesystems
                # Only call this function with verify turned on when it's needed (e.g. when open_mfdataset fails).
                if os.path.exists(file_pat):
                    job_files.append(file_pat)
                else:
                    if fdate.month == 10 and fdate.day == 1 and fdate.hour == 0:
                        # Sometimes a dataset includes the first hour of the next water year in
                        # each water year directory; we need this file.
                        wy_dir = f'WY{cdate.year}'
                        file_pat = eval(f"f'{file_pattern}'")

                        if os.path.exists(file_pat):
                            job_files.append(file_pat)

                        break
            else:
                job_files.append(file_pat)
    return job_files


def delete_dir(fs, path):
    """
    Recursively remove directory using fsspec
    """
    try:
        fs.rm(path, recursive=True)
    except FileNotFoundError:
        pass


def get_maxmem_per_thread(client, max_percent=0.7, verbose=False):
    """
    Returns the maximum amount of memory to use per thread for chunking.
    """

    # client: dask client
    # max_percent: Maximum percentage of memory to use for rechunking per thread

    # Max total memory in gigabytes for cluster
    total_mem = sum(vv['memory_limit'] for vv in client.scheduler_info()['workers'].values()) / 1024**3
    total_threads = sum(vv['nthreads'] for vv in client.scheduler_info()['workers'].values())

    if verbose:
        print('-'*60)
        print(f'Total memory: {total_mem:0.1f} GB; Threads: {total_threads}')

    max_mem = f'{total_mem / total_threads * max_percent:0.0f}GB'

    if verbose:
        print(f'Maximum memory per thread for rechunking: {max_mem}')
        print('-'*60)

    return max_mem


def read_metadata(filename):
    """
    Read the metadata information file
    """

    use_cols = ['varname', 'long_name', 'integration_length', 'description',
                'notes', 'units', 'scale_factor', 'valid_range', 'flag_values',
                'flag_meanings', 'coordinates']

    coord_map = {'XLONG XLAT': 'lon lat',
                 'XLONG XLAT XTIME': 'lon lat',
                 'XLONG_U XLAT_U': 'lon_u lat_u',
                 'XLONG_U XLAT_U XTIME': 'lon_u lat_u',
                 'XLONG_V XLAT_V': 'lon_v lat_v',
                 'XLONG_V XLAT_V XTIME': 'lon_v lat_v'}

    df = pd.read_csv(filename, sep='\t', index_col='varname', usecols=use_cols)

    # For example, when doing 'df[col].method(value, inplace=True)',
    # try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to
    # perform the operation inplace on the original object.

    # Change the coordinates to match the new lon/lat variable names
    for kk, vv in coord_map.items():
        df['coordinates'].mask(df['coordinates'] == kk, vv, inplace=True)

    # Add a few empty attributes, in the future these may already exist
    df['grid_mapping'] = pd.NA
    df['axis'] = pd.NA
    df['standard_name'] = pd.NA

    # Set grid_mapping attribute for variables with non-empty coordinates attribute
    df['grid_mapping'].mask(df['coordinates'].notnull(), 'crs', inplace=True)

    # Rename the XTIME variable to time
    df = df.rename(index={'XTIME': 'time'})
    df.loc['time', 'axis'] = 'T'
    df.loc['time', 'standard_name'] = 'time'

    # Rechunking will crash if units for 'time' is overridden with an error
    # like the following:
    # ValueError: failed to prevent overwriting existing key units in attrs
    df.loc['time', 'units'] = pd.NA

    return df


def read_metadata_xtrm(filename):
    """
    Read the metadata information file
    """

    use_cols = ['varname', 'long_name', 'description',
                'units', 'coordinates']

    coord_map = {'XLONG XLAT XTIME': 'lon lat'}

    df = pd.read_csv(filename, sep='\t', index_col='varname', usecols=use_cols)

    # For example, when doing 'df[col].method(value, inplace=True)',
    # try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to
    # perform the operation inplace on the original object.

    # Change the coordinates to match the new lon/lat variable names
    for kk, vv in coord_map.items():
        df['coordinates'].mask(df['coordinates'] == kk, vv, inplace=True)

    # Add a few empty attributes, in the future these may already exist
    df['grid_mapping'] = pd.NA
    df['axis'] = pd.NA
    df['standard_name'] = pd.NA

    # Set grid_mapping attribute for variables with non-empty coordinates attribute
    df['grid_mapping'].mask(df['coordinates'].notnull(), 'crs', inplace=True)

    # Add a time variable
    df.loc['time'] = dict(axis='T', standard_name='time', long_name='time', units=pd.NA)

    return df


def rechunker_wrapper(source_store, target_store, temp_store, chunks=None,
                      mem=None, consolidated=False, verbose=True):

    t1 = time.time()

    if isinstance(source_store, xr.Dataset):
        g = source_store  # Work directly with a dataset
        ds_chunk = g
    else:
        g = zarr.group(str(source_store))
        # Get the correct shape from loading the store as xr.dataset and parse the chunks
        ds_chunk = xr.open_zarr(str(source_store))

    group_chunks = {}

    # Newer tuple version that also takes into account when specified chunks are larger than the array
    for var in ds_chunk.variables:
        # Pick appropriate chunks from above, and default to full length chunks for
        # dimensions that are not in `chunks` above.
        group_chunks[var] = []

        for di in ds_chunk[var].dims:
            if di in chunks.keys():
                if chunks[di] > len(ds_chunk[di]):
                    group_chunks[var].append(len(ds_chunk[di]))
                else:
                    group_chunks[var].append(chunks[di])

            else:
                group_chunks[var].append(len(ds_chunk[di]))

        group_chunks[var] = tuple(group_chunks[var])

    if verbose:
        print(f"Rechunking to: {group_chunks}")
        print(f"Memory: {mem}")

    rechunked = rechunker.rechunk(g, target_chunks=group_chunks, max_mem=mem,
                                  target_store=target_store, temp_store=temp_store)
    rechunked.execute(retries=10)

    if consolidated:
        if verbose:
            print('Consolidating metadata')

        zarr.convenience.consolidate_metadata(target_store)

    if verbose:
        print(f'    rechunker: {time.time() - t1:0.3f} s')


def set_file_path(path1, path2=None):
    """Helper function to check/set the full path to a file.
       path1 should include a filename
       path2 should only be a directory
    """

    if os.path.isfile(path1):
        # File exists, use the supplied path
        file_path = os.path.realpath(path1)
    else:
        # File does not exist
        if path2:
            file_path = os.path.realpath(f'{path2}/{path1}')

            if os.path.isfile(file_path):
                # File exists in path2 so append it to the path
                print(f'Using filepath, {file_path}')
            else:
                raise FileNotFoundError(f'File, {path1}, does not exist in {path2}')
        else:
            raise FileNotFoundError(f'File, {path1}, does not exist')

    return file_path


def set_target_path(path, base_dir=None, verbose=False):

    if os.path.isdir(path):
        # We're good, use it
        new_path = os.path.realpath(path)
        if verbose:
            print(f'{new_path} exists.')
    else:
        # Path is not a directory, does the parent directory exist?
        pdir = os.path.dirname(path)

        if pdir == '':
            # There is no parent directory, try using base_dir if it exists
            if base_dir:
                # create/use
                if not os.path.isdir(base_dir):
                    raise FileNotFoundError(f'Base directory, {base_dir}, does not exist')

                new_path = f'{base_dir}/{path}'

                if os.path.isdir(new_path):
                    if verbose:
                        print(f'Using existing target path, {new_path}')
                else:
                    os.mkdir(new_path)

                    if verbose:
                        print(f'Creating target relative to base directory, {new_path}')
            else:
                # No base_dir supplied; create target path in current directory
                new_path = os.path.realpath(path)
                os.mkdir(new_path)

                if verbose:
                    print(f'Target path, {new_path}, created')
        elif os.path.isdir(pdir):
            # Parent path exists we just need to create the child directory
            new_path = os.path.realpath(path)
            os.mkdir(new_path)

            if verbose:
                print(f'Parent, {pdir}, exists. Created {new_path} directory')
        else:
            raise FileNotFoundError(f'Parent, {pdir}, of target path, {path}, does not exist')

    return new_path
