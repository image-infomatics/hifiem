#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:22:27 2022

@author: ykreinin
"""

import os, sys, glob, argparse as argp, tqdm, numpy as np, PIL, skimage, h5py, ast, time, configparser
from skimage.io import imread, imsave
from datetime import datetime

import multiprocessing as mp, traceback
from multiprocessing import shared_memory
from functools import partial

PIL.Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------------------------------------------------
class struct:
    def __getitem__(self, item):
        return self.__dict__[item] #self[item]

    def __setitem__(self, item, val):
        self.__dict__[item] = val

    def __repr__(self):
        return self.__dict__.__repr__()

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def isfield(self, name: str, checkNone: bool = True) -> bool:
        return name in self.__dict__ and (not checkNone or (not self[name] is None))

# ------------------------------------------------------------------------------------------------------
def save_dataset(hfile, name, data, **kwargs):
    if name in hfile.keys():
        del hfile[name]
    hfile.create_dataset(name, data = data, **kwargs)

# ------------------------------------------------------------------------------------------------------
def get_config_val(config, section, name, **kwargs):
    try: return config.getboolean(section, name, **kwargs)
    except: pass
    try: return config.getint(section, name, **kwargs)
    except: pass
    try: return config.getfloat(section, name, **kwargs)
    except: s = config.get(section, name, **kwargs)
    try: return ast.literal_eval(s)
    except: return s

# ------------------------------------------------------------------------------------------------------
def get_config_literal(config, section, name, fallback = None):
    try: return ast.literal_eval(config.get(section, name))
    except: return fallback

# ------------------------------------------------------------------------------------------------------
def read_config_section(config, SECTION, output = None, no_defaults = True, as_struct = False):
    if (SECTION is None) or (not SECTION in config):
        return None
    if not output is None:
        as_struct = not isinstance(output, dict)
    else:
        output = struct() if as_struct else dict()

    for name in config[SECTION]:
        if no_defaults and (name in config.defaults()):
            continue
        elif as_struct:
            setattr(output, name, get_config_val(config, SECTION, name))
        else:
            output[name] = get_config_val(config, SECTION, name)

    return output

# ------------------------------------------------------------------------------------------------------
def timedelta_to_string(delta):
    if delta < 1:
        return f"{1000 * delta:.0f} msecs"
    elif delta < 5:
        return f"{delta : 1.3f} secs"

    seconds = int(delta)
    if seconds <= 60:
        return f"{seconds:02d} secs"

    minutes  = seconds // 60
    seconds %= 60
    if minutes <= 60:
        return f"{minutes:02d} mins {seconds:02d} secs"

    hours = minutes // 60
    minutes %= 60
    return f"{hours:02d} hours {minutes:02d} mins"

# ------------------------------------------------------------------------------------------------------
def PerformanceTime(text, start):
    return f"{text} - {timedelta_to_string(time.time() - start)}"

# ------------------------------------------------------------------------------------------------------
class time_it:
    def __init__(self, message = "", print_fn = print, print_exception = True, **kwargs):
        self._message  = message
        self._print_fn = print_fn
        self._print_exception = print_exception
        self._time  = 0.0
        self.kwargs = kwargs

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def print(self, *args, timedelta = None, **kwargs):
        if timedelta is None: timedelta = self._time
        self._print_fn(*args, "completed in", timedelta_to_string(timedelta), **kwargs)

    def print_total(self, *args, **kwargs):
        self._time = time.monotonic() - self._start
        self.print(*args, **kwargs)

    def print_progress(self, *args, **kwargs):
        _time, self._time = self._time, time.monotonic() - self._start
        _time = self._time - _time
        self.print(*args, timedelta = _time, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        self._time = time.monotonic() - self._start
        if exc_type is None:
            self.print(self._message, **self.kwargs)
        elif self._print_exception:
            self._print_fn(f"{self._message} failed with {exc_value}")

# ------------------------------------------------------------------------------------------------------
def read_config_file(filename = os.path.join(".", "config.ini"), default_section = "common"):
    config = configparser.RawConfigParser(default_section = default_section, inline_comment_prefixes = ("#",)) #, interpolation=None)
    config.optionxform = lambda x: x
    config.BOOLEAN_STATES = { "Yes" : True, "No" : False, "yes" : True, "no" : False, "True" : True, "False" : False, "true" : True, "false" : False }
    config.read(filename)
    return config

# ------------------------------------------------------------------------------------------------------
def cv_keypoints_to_struct(kpoints):
    count = len(kpoints)
    kpts = struct(x = np.zeros(shape = count, dtype = np.float32),
                  y = np.zeros(shape = count, dtype = np.float32),
                  angle = np.zeros(shape = count, dtype = np.float32),
                  size  = np.zeros(shape = count, dtype = np.float32),
                  response = np.zeros(shape = count, dtype = np.float32),
                  class_id = np.zeros(shape = count, dtype = np.int32),
                  octave   = np.zeros(shape = count, dtype = np.int32))

    for index, kp in enumerate(kpoints):
        kpts.x[index] = kp.pt[0]
        kpts.y[index] = kp.pt[1]
        kpts.angle[index] = kp.angle
        kpts.size[index]  = kp.size
        kpts.response[index]  = kp.response
        kpts.class_id[index] = kp.class_id
        kpts.octave[index]   = kp.octave

    return kpts

#===================================================================================================================
def safe_sqrt(x, print_fn = None):
    bwmask = x < 0
    negative = np.count_nonzero(bwmask)
    if negative == 0:
        return np.sqrt(x)

    if print_fn: print_fn(f"\t{negative} elements are below zero, minval = {np.sqrt(-np.min(x[bwmask])) :.2f}")
    bwmask = (~bwmask).astype(x.dtype)
    return np.sqrt(x * bwmask)

#===================================================================================================================
def load_polymodel(filename):
    if filename is None:
        return None

    with h5py.File(filename, mode = "r") as hfile:
        polymodel = struct(coeffs = hfile["Coefficients"][:], term = hfile["ModelTerms"][:])
        polymodel.term = polymodel.term[::-1, :]

    return polymodel

#===================================================================================================================
def correct_section_background(image, shape, intensity_shift = None, polymodel = None, pos = (0, 0, 0)):
    output = np.array(image, dtype = np.float32)

    bwflat = np.isnan(output)
    if np.all(bwflat):
        return output, bwflat

    if not polymodel is None:
        mask   = polymodel.term.any(axis = 0) # ignore constant offset coefficient
        coeffs = polymodel.coeffs[mask]
        term   = polymodel.term[:, mask]

        def apply_polymodel():
            # nonlocal zyx
            zyx = np.unravel_index(index[start:end], output.shape)
            zyx = (pos[0] / (shape[0] - 1), (pos[1] + zyx[0]) / (shape[1] - 1), (pos[2] + zyx[1]) / (shape[2] - 1))

            for term_idx in range(term.shape[1]):
                vals = np.ones((end - start,), dtype = np.float32)
                for dim_idx in range(3):
                    if term[dim_idx, term_idx] == 0:
                        continue
                    elif term[dim_idx, term_idx] == 1:
                        vals *= zyx[dim_idx]
                    else:
                        vals *= zyx[dim_idx] ** term[dim_idx, term_idx]

                output.flat[index[start:end]] -= coeffs[term_idx] * vals
    elif (not intensity_shift is None) and (intensity_shift != 0):
        def apply_polymodel(): return
    else:
        return output, bwflat

    index  = np.flatnonzero(~bwflat)
    start  = 0
    while start < index.size:
        end = min(index.size, start + 500000)
        # zyx = np.unravel_index(index[start:end], output.shape)
        if (not intensity_shift is None) and (intensity_shift != 0):
            output.flat[index[start:end]] -= intensity_shift

        apply_polymodel()
        start = end

    return output, bwflat

#===================================================================================================================
def prepare_slurm_array_task(num_of_chunks, return_node_idx = False):
    SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT"))
    SLURM_ARRAY_TASK_ID  = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    SLURM_ARRAY_TASK_MIN = int(os.getenv("SLURM_ARRAY_TASK_MIN"))
    # SLURM_CPUS_PER_TASK  = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    SLURM_PROCID = int(os.getenv("SLURM_PROCID"))
    SLURM_NTASKS = int(os.getenv("SLURM_NTASKS")) # number of tasks (processes) per ARRAY TASK

    # balanced solution, extra chunks are spread between first tasks
    num_of_tasks = SLURM_NTASKS * SLURM_ARRAY_TASK_COUNT
    chunks_per_task, num_of_extra_sections = divmod(num_of_chunks, num_of_tasks)
    node_index = (SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN) * SLURM_NTASKS + SLURM_PROCID
    start_chunk_idx = node_index * chunks_per_task + min(node_index, num_of_extra_sections)
    chunks_per_task += node_index < num_of_extra_sections
    if return_node_idx:
        return start_chunk_idx, chunks_per_task, node_index
    else:
        return start_chunk_idx, chunks_per_task


#===================================================================================================================
def realign_section(source, dx, dy, shape = None):
    #     source_x = max(0, dx)
    #     target_x = max(0, -dx)
    #     # x_len = min(shape[1], section.shape[1] - dx)
    #     source_end = min(shape[1] + dx, section.shape[1])
    #     target_end = min(shape[1], section.shape[1] - dx)

    if shape is None: shape = source.shape
    offset = (dy, dx)
    def source_slice(axis):
        return slice(max(0, offset[axis]), min(shape[axis] + offset[axis], source.shape[axis]))

    def target_slice(axis):
        return slice(max(0, -offset[axis]), min(shape[axis], source.shape[axis] - offset[axis]))

    target = np.full(shape, np.nan, dtype = np.float32)
    target[target_slice(0), target_slice(1)] = source[source_slice(0), source_slice(1)]
    return target

# ========================================================================================================================
def create_symlinks(args):
    source = os.path.realpath(args.source)
    target = os.path.realpath(args.target)

    print("source: ", source)
    print("destination folder", target)

    os.makedirs(target, exist_ok = True)

    ext = os.path.splitext(source)[1]
    target = os.path.join(target, "{file_idx:05d}" + ext)

    filelist = sorted(glob.iglob(source, recursive = args.recursive))
    for file_idx in tqdm.trange(len(filelist)):
        filename = filelist[file_idx]
        os.symlink(filename, target.format(file_idx = file_idx))


# ========================================================================================================================
def patch_from_image(image, tinfo):
    if tinfo.flipud: image = np.flipud(image)
    if tinfo.fliplr: image = np.fliplr(image)
    image = image[tinfo.chunk_start[1] : tinfo.chunk_start[1] + tinfo.chunk_shape[1],
                  tinfo.chunk_start[2] : tinfo.chunk_start[2] + tinfo.chunk_shape[2]]
    if tinfo.invert and np.issubdtype(image.dtype, np.integer):
        image = np.iinfo(image.dtype).max - image
    return image

# ========================================================================================================================
def read_patch_worker(tinfo, section_idx):
    try:
        filename = tinfo.filelist[section_idx] if tinfo.isfield("filelist") else tinfo.input.format(slice_idx = section_idx)
        image = patch_from_image(imread(filename), tinfo)
        shmem = shared_memory.SharedMemory(name = tinfo.shmem, create = False)
        chunk = np.ndarray(shape = tinfo.chunk_shape, dtype = image.dtype, buffer = shmem.buf)
        chunk[section_idx - tinfo.chunk_start[0], ...] = image
        shmem.close()
        return section_idx, None
    except Exception as ex:
        return section_idx, str(ex)


# ========================================================================================================================
def create_chunk(source, zyx, shape, workers = 8, invert = False, flipud = False, fliplr = False):

    tinfo = struct(filelist = sorted(glob.iglob(os.path.join(source, "*.*"))), invert = invert, fliplr = fliplr, flipud = flipud)
    image = imread(tinfo.filelist[zyx[0]])
    tinfo.chunk_start = zyx
    tinfo.chunk_shape = (min(shape[0], len(tinfo.filelist)  - zyx[0]),
                         min(shape[1], image.shape[0] - zyx[1]),
                         min(shape[2], image.shape[1] - zyx[2]))
    try:
        with time_it(f"extract chunk at {zyx=}"):
            shmem = shared_memory.SharedMemory(create = True, size = image.itemsize * np.prod(tinfo.chunk_shape))
            chunk = np.ndarray(shape = tinfo.chunk_shape, dtype = image.dtype, buffer = shmem.buf)
            chunk[0, ...] = patch_from_image(image, tinfo)
            tinfo.shmem = shmem.name

            max_tasks_per_child = (tinfo.chunk_shape[0] + workers - 2) // workers # we have already processed one section
            print(f"start process pool of {workers} workers, {max_tasks_per_child=}, {os.cpu_count()} CPUs on host")
            with mp.Pool(processes = workers, maxtasksperchild = max_tasks_per_child) as pool:
                print("Process pool has started")
                for section_idx, text in pool.imap_unordered(partial(read_patch_worker, tinfo), range(tinfo.chunk_start[0] + 1, tinfo.chunk_start[0] + tinfo.chunk_shape[0])):
                    print(f"section {section_idx:05d}", "is completed" if text is None else f"failed with {text}")
        return chunk.copy()

    except Exception as ex:
        print(f"Main process failed with {ex}")
        return None

    finally:
        shmem.close()

# ========================================================================================================================
def save_patch_worker(tinfo, section_idx):
    try:
        filename = tinfo.filelist[section_idx] if tinfo.isfield("filelist") else tinfo.input.format(slice_idx = section_idx)
        imsave(os.path.join(tinfo.output, f"{section_idx :05d}.tif"), patch_from_image(imread(filename), tinfo), check_contrast = False, compression = "zlib")
    except: # Exception as ex:
        traceback.print_exc()
    return section_idx

# ========================================================================================================================
def _create_chunk(args):
    os.makedirs(os.path.split(args.prefix)[0], exist_ok = True)

    z = args.start[0]
    y = args.start[1]
    x = args.start[2]
    filename = f"{args.prefix}{z:05d}_{y:05d}_{x:05d}_{args.size[0]}_{args.size[1]}_{args.size[2]}"

    if args.tif:
        os.makedirs(filename, exist_ok = True)
        tinfo = struct(output = filename, invert = args.invert, flipud = args.flipud, fliplr = args.fliplr)
        if args.index:
            tinfo.input = os.path.join(args.source, "{slice_idx:05d}.tif")
            section_count = np.iinfo(np.uint16).max
            input_file = tinfo.input.format(slice_idx = z)
        else:
            tinfo.filelist = sorted(glob.iglob(os.path.join(args.source, "*.*")))
            section_count  = len(tinfo.filelist)
            input_file = tinfo.filelist[z]

        image = imread(input_file)
        tinfo.chunk_start = args.start
        tinfo.chunk_shape = (min(args.size[0], section_count - z),
                             min(args.size[1], image.shape[0] - y),
                             min(args.size[2], image.shape[1] - x))

        imsave(os.path.join(tinfo.output, f"{z :05d}.tif"), patch_from_image(image, tinfo), check_contrast = False, compression = "zlib")
        try:
            with time_it(f"extract chunk at {tinfo.chunk_start}"):
                workers = args.workers
                max_tasks_per_child = (tinfo.chunk_shape[0] + workers - 2) // workers # we have already processed one section
                print(f"start process pool of {workers} workers, {max_tasks_per_child=}, {os.cpu_count()} CPUs on host")
                with mp.Pool(processes = workers, maxtasksperchild = max_tasks_per_child) as pool:
                    for section_idx in pool.imap_unordered(partial(save_patch_worker, tinfo), range(z + 1, z + tinfo.chunk_shape[0])):
                        print(f"section {section_idx:05d}", "is completed")
        except Exception as ex:
            print(f"Main process failed with {ex}")
    else:
        chunk = create_chunk(args.source, args.start, args.size, args.workers, invert = args.invert, flipud = args.flipud, fliplr = args.fliplr)
        with h5py.File(filename + ".h5", mode = "a") as hfile:
            save_dataset(hfile, name = args.dataset, data = chunk, compression = "gzip", chunks = (1, *chunk.shape[1:]))

# ========================================================================================================================
def _test_read(args):
    # start = time.time()
    filelist = sorted(glob.iglob(os.path.join(args.source, "*.*")))[args.start:]
    if not args.count is None:
        filelist = filelist[:args.count]

    for filename in tqdm.tqdm(filelist):
        imread(filename)

# ---- main section
if __name__ == "__main__":
    parser = argp.ArgumentParser(description = "script file contains commands and helpers to operate with waspem data")

    subparsers = parser.add_subparsers()
    # symlink  "/media/ykreinin/My Book/Flatiron Waspem/Sample-1/raw_inlens_tif_files/*/*.tif" "/media/ykreinin/My Book/Flatiron Waspem/Sample-1/raw_ordered" --recursive
    symlink_parser = subparsers.add_parser('symlink', description = "enumerate files in source folder and creates numeric symlinks in destination folder")
    symlink_parser.add_argument("source", type = str, help = "folder and file mask of the source files to be enumerated by symlinks")
    symlink_parser.add_argument("target", type = str, default = ".", help = "folder to be used for symlinks")
    symlink_parser.add_argument("-r", "--recursive", default = False, action = "store_true", help = "search inside sub-folders")
    symlink_parser.set_defaults(func = create_symlinks)

    # srun --partition=ccn -n 1 python ../../plugins/fltlib.py chunk ./tiff ./chunks/aligned -s 4014 4146 5384
    chunk_parser = subparsers.add_parser("chunk", description = "cut a chunk of data from sequence of sections")
    chunk_parser.add_argument("source", type = str, help = "folder containing stack as a sequence of images")
    chunk_parser.add_argument("prefix", type = str, help = "prefix of path and filename to be used to save a chunk")
    chunk_parser.add_argument("-s", "--start", type = int, nargs = 3, help = "specifies start location of chunk zyx")
    chunk_parser.add_argument("-c", "--size", type = int, nargs = 3, default = [512, 512, 512], help = "specifies size of chunk zyx")
    chunk_parser.add_argument("-t", "--tif", default = False, action = "store_true", help = "specifies that chunk should be saved in tif file, otherwise .h5 file is created")
    chunk_parser.add_argument("-ds", "--dataset", type = str, default = "vals", help = "specifies the name of dataset in .h5 file")
    chunk_parser.add_argument("-w", "--workers", type = int, default = 8, help = "number of worker threads to be used for parallel read")
    chunk_parser.add_argument("-i", "--invert", default = False, action = "store_true", help = "specifies that image has to be inverted")
    chunk_parser.add_argument("-ud", "--flipud", default = False, action = "store_true", help = "flips image updown")
    chunk_parser.add_argument("-lr", "--fliplr", default = False, action = "store_true", help = "flips image leftright")
    chunk_parser.add_argument("-idx", "--index", default = False, action = "store_true", help = "use section index to construct filename")
    # chunk_parser.add_argument("-z", "--gzip", default = False, action = "store_true", help = "specifies that data should be compressed")
    chunk_parser.set_defaults(func = _create_chunk)

    test_read = subparsers.add_parser("test-read", description = "reads a number of sections to the memory in order to test imread speed")
    test_read.add_argument("source", type = str, help = "folder containing stack as a sequence of images")
    test_read.add_argument("-s", "--start", type = int, default = 0, help = "specifies the first section")
    test_read.add_argument("-c", "--count", type = int, default = None, help = "specifies the number of sections to read")
    test_read.set_defaults(func = _test_read)

    args = parser.parse_args()
    args.func(args)
