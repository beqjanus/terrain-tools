#!/usr/bin/env python
""" Create a SL-style raw file from high-resolution terrain data in CSV format.
This script reads a CSV file containing (X, Y, Z) triples, builds a high-resolution grid,
downsamples it to a lower resolution, and encodes the heights into a 13-channel raw file.
The output is suitable for use in Second Life or OpenSim.
Not currently tested for sizes other than 256x256.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# CONFIG defaults and ARGS
global args, resolution, region_name

CSV_FILE    = 'highres.csv'
RAW_OUTFILE = 'terrain.raw'
STEP        = 1.0    # metres for high‑res sampling
FACTOR      = 1        # downsample factor 
WATER_LEVEL = 20.0      # default water level in metres
TARGET_DIM  = 256       # SL terrain is always 256x256

def compare_and_plot_error(raw_path: str, hires_grid: np.ndarray, target_dim: int = 256):
    """
    Load an SL‐style raw file and compare its decoded heights to a reference grid.
    Computes standard deviation of the percentage error and displays a histogram.
    """
    # Load raw
    data = np.fromfile(raw_path, dtype=np.uint8)
    expected = target_dim * target_dim * 13
    if data.size != expected:
        raise ValueError(f"Expected {expected} bytes, got {data.size}")
    raw_arr = data.reshape((target_dim, target_dim, 13))
    R = raw_arr[..., 0].astype(np.float32)
    G = raw_arr[..., 1].astype(np.float32)
    raw_heights = (R * G) / 128.0

    # Compute error percentage, but skip where hires_grid == 0
    mask = hires_grid != 0
    err_pct = np.full_like(hires_grid, np.nan, dtype=np.float32)
    err_pct[mask] = (raw_heights[mask] - hires_grid[mask]) / hires_grid[mask] * 100.0

    # Stats
    std_pct = np.nanstd(err_pct)
    mean_pct = np.nanmean(err_pct)
    print(f"Mean error: {mean_pct:.3f}%")
    print(f"Std dev of error: {std_pct:.3f}%")

    # Histogram (ignore nan values)
    plt.figure()
    plt.hist(err_pct[np.isfinite(err_pct)].flatten(), bins=50)
    plt.title("Error Percentage Distribution")
    plt.xlabel("Error (%)")
    plt.ylabel("Frequency")
    plt.show() 

def compare_raw_to_hires(raw_path: str,
                         hires_grid: np.ndarray,
                         target_dim: int = 256) -> dict:
    """
    Compare the height range in an existing SL‐style raw file to a high‑res grid.

    Parameters
    ----------
    raw_path : str
        Path to the interleaved 13‑channel 8‑bit .raw (256x256x13).
    hires_grid : np.ndarray
        High‑res or downsampled 2D array of shape (256,256) containing floats.
    target_dim : int
        Expected dimension of the raw file (default 256).

    Returns
    -------
    dict
      {
        'hires_min': float, 'hires_max': float,
        'raw_min': float,  'raw_max':  float,
        'delta_min': float,'delta_max': float
      }
      where delta = raw – hires.
    """
    
    # Load raw file
    data = np.fromfile(raw_path, dtype=np.uint8)
    expected_bytes = target_dim * target_dim * 13
    if data.size != expected_bytes:
        raise ValueError(f"File has {data.size} bytes, expected {expected_bytes}.")
    raw_arr = data.reshape((target_dim, target_dim, 13))

    # Decode heights from R & G channels
    R = raw_arr[:, :, 0].astype(np.float32)
    G = raw_arr[:, :, 1].astype(np.float32)
    raw_heights = (R * G) / 128.0

    # Compute ranges
    hires_min, hires_max = np.nanmin(hires_grid), np.nanmax(hires_grid)
    raw_min,   raw_max   = np.nanmin(raw_heights), np.nanmax(raw_heights)

    return {
        'hires_min': hires_min,
        'hires_max': hires_max,
        'raw_min':   raw_min,
        'raw_max':   raw_max,
        'delta_min': raw_min  - hires_min,
        'delta_max': raw_max  - hires_max
    }

def read_csv_header(csv_file):
    """Read the first line of the CSV file and return region name and resolution."""
    with open(csv_file, 'r') as f:
        header = f.readline().strip().split(',')
        if len(header) < 2:
            raise ValueError("CSV header must contain at least two fields: region_name and resolution.")
        region_name = header[0]
        resolution = header[1]
    return region_name, float(resolution)

def load_hires_csv(csv_file):
    """Load high-res CSV file and return (X, Y, Z) triples."""
    triples = []
    with open(csv_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # Skip first 3 fields and drop empties
            raw = [p for p in parts[3:] if p]
            vals = list(map(float, raw))
            for i in range(0, len(vals), 3):
                x, y, z = vals[i:i+3]
                triples.append((x, y, z))
    if not triples:
        raise ValueError("No valid XYZ data found in CSV.")
    return np.array(triples)

def normalise_coordinates(triples):
    """Normalise coordinates so minimum X,Y is (0,0)."""
    x_min, y_min = triples[:,0].min(), triples[:,1].min()
    triples[:,0] -= x_min
    triples[:,1] -= y_min
    return triples

def build_high_res_grid(triples, resolution):
    """Build a high-resolution grid from (X, Y, Z) triples at the given resolution."""
    # Scale coordinates to grid indices
    xi = np.round(triples[:, 0] / resolution).astype(int)
    yi = np.round(triples[:, 1] / resolution).astype(int)
    x_max = xi.max()
    y_max = yi.max()
    print_if_not_quiet(f"Building high-res grid of {y_max+1} rows x {x_max+1} cols at resolution {resolution}m")
    nx = x_max + 1
    ny = y_max + 1
    Z_hi = np.full((ny, nx), np.nan, dtype=float)
    Z_hi[yi, xi] = triples[:, 2]
    return Z_hi

def report_missing(Z_hi):
    """Report missing values in the high-res grid."""
    total = Z_hi.size
    missing = np.isnan(Z_hi).sum()
    print_if_not_quiet(f"High-res grid: {Z_hi.shape[0]} rows x {Z_hi.shape[1]} cols, "
                       f"missing {missing}/{total} ({missing/total*100:.2f}%)")
    if missing:
        print_if_not_quiet("  (consider filling missing before proceeding)")
    return missing

def check_dimensions_and_trim(Z_hi, target_dim=TARGET_DIM):
    """Check if the grid dimensions are at least TARGET_DIM*FACTOR and trim excess"""
    ny, nx = Z_hi.shape
    if ny < target_dim or nx < target_dim:
        raise RuntimeError(f"Grid {ny}x{nx} smaller than {target_dim}x{target_dim}")
    if (ny, nx) != (target_dim, target_dim):
        print_if_not_quiet(f"Trimming grid from {ny}x{nx} to {target_dim}x{target_dim}")
    Z_hi = Z_hi[:target_dim, :target_dim]
    return Z_hi

def downsample_grid_to_step_size(Z_hi, factor=FACTOR, method='nearest'):
    """Downsample the high-res grid by a factor of factor.
    method: 'nearest' (default) or 'average'
    """
    if factor <= 1:
        return Z_hi
    if Z_hi.shape[0] % factor != 0 or Z_hi.shape[1] % factor != 0:
        raise ValueError(f"Grid dimensions {Z_hi.shape} not divisible by factor {factor}")
    if method == 'nearest':
        return Z_hi[::factor, ::factor]
    elif method == 'average':
        ny, nx = Z_hi.shape
        Z_hi_reshaped = Z_hi.reshape(ny // factor, factor, nx // factor, factor)
        return np.nanmean(Z_hi_reshaped, axis=(1, 3))
    else:
        raise ValueError(f"Unknown downsampling method: {method}")

def encode_height(h):
    # Fast path for zero height
    if np.isnan(h) or h == 0.0:
        return np.uint8(0), np.uint8(1)
    best = (0, 0, float('inf'))
    for R in range(255, 0, -1):  # Loop from 255 down to 1
        G = int(round(h * 128 / R))
        if 0 <= G < 256:
            err = abs(R * G / 128 - h)
            # Prefer lower G if error is the same
            if err < best[2] or (err == best[2] and G < best[1]):
                best = (R, G, err)
    return np.uint8(best[0]), np.uint8(best[1])

def encode_height_quantized(h, quanta_per_m=64):
    """
    Encode height h into R,G bytes such that reconstructed height R*G/128
    always falls on the 1/quanta_per_m grid by quantizing G channel.
    """
    # Fast path for zero height
    if np.isnan(h) or h == 0.0:
        return np.uint8(0), np.uint8(1)
    
    M = 128 // quanta_per_m  # G must be multiple of M, here 2 for quanta_per_m=64
    best = (0, 0, float('inf'))
    
    for R in range(255, 0, -1):
        G_ideal = h * 128 / R
        G_round = int(round(G_ideal / M)) * M
        if 0 <= G_round < 256:
            err = abs(R * G_round / 128.0 - h)
            # tie-break with lower G
            if err < best[2] or (err == best[2] and G_round < best[1]):
                best = (R, G_round, err)
    
    # Fallback to original if nothing found
    if best[2] == float('inf'):
        for R in range(255, 0, -1):
            G = int(round(h * 128 / R))
            err = abs(R * G / 128.0 - h)
            if err < best[2] or (err == best[2] and G < best[1]):
                best = (R, G, err)
    
    return np.uint8(best[0]), np.uint8(best[1])

def create_blank_terrain(target_dim=TARGET_DIM):
    # create a hi_res array of all zeroes ready for processing. Using this we can later increase the height to create arbitrary level fields
    """Create a blank terrain grid of zeros."""
    if target_dim <= 0:
        raise ValueError("Target dimension must be positive")
    return np.zeros((target_dim, target_dim), dtype=np.float32)

def create_ramp_terrain(max_height, target_dim=TARGET_DIM):
    """
    Create a ramp terrain grid from height 0 (left, X=0) to max_height (right, X=max).
    The ramp increases linearly along the X axis.
    """
    if target_dim <= 0:
        raise ValueError("Target dimension must be positive")
    # Each row is a linear ramp from 0 to max_height
    ramp = np.linspace(0, max_height, target_dim, dtype=np.float32)
    return np.tile(ramp, (target_dim, 1))

def build_terrain_array(height_chan, height_unit_chan, water_byte, target_dim=TARGET_DIM):
    """Build the final 13-channel terrain array.
    Use Row major order:
    - Channel 0: Height (R)
    - Channel 1: Height unit (G)
    - Channel 2: Water (B)
    - Channels 3-12: Unused (zeros)
    13 channels total.
    When np_array.to_bytes() is called, it will be written in row-major order.
    Giving interleaved output
    """
    if height_chan.shape != (target_dim, target_dim):
        raise ValueError(f"Height channel shape {height_chan.shape} does not match target dimension {target_dim}")
    if height_unit_chan.shape != (target_dim, target_dim):
        raise ValueError(f"Height unit channel shape {height_unit_chan.shape} does not match target dimension {target_dim}")
    
    # Create the water channel and unused channels
    water_channel = np.full((target_dim, target_dim), water_byte, dtype=np.uint8)
    unused = np.zeros((target_dim, target_dim, 10), dtype=np.uint8)
    
    # Stack the channels
    out_arr = np.dstack((height_chan, height_unit_chan, water_channel, unused))
    return out_arr

def write_output_raw(out_arr, target_dim=TARGET_DIM):
    """Write the output raw file."""
    if out_arr.shape != (target_dim, target_dim, 13):
        raise ValueError(f"Output array shape {out_arr.shape} does not match target dimension {target_dim}")
    
    # Ensure the output directory exists
    if os.path.dirname(args.raw_out) != "":
        os.makedirs(os.path.dirname(args.raw_out), exist_ok=True)
    
    # Write the raw file
    expected_bytes = target_dim * target_dim * 13
    print_if_not_quiet(f"Writing {args.raw_out}: {expected_bytes} bytes")
    if out_arr.dtype != np.uint8:
        raise ValueError(f"Output array dtype {out_arr.dtype} must be np.uint8")
    if out_arr.nbytes != expected_bytes:
        raise ValueError(f"Output array size {out_arr.nbytes} bytes does not match expected {expected_bytes} bytes")

    with open(args.raw_out, 'wb') as f:
        f.write(out_arr.tobytes())

    print_if_not_quiet(f"Wrote {args.raw_out} ({TARGET_DIM}x{TARGET_DIM}x13)")

def print_if_not_quiet(*messages, **kwargs):
    """Print only if not in quiet mode."""
    if not args.quiet:
        print(*messages, **kwargs)

def main():
    """ check arguments. Allow 
    --quiet (no prompts) 
    --help (usage info) 
    --compare-with <raw_file> (external older raw file to compare the hi-res data with) 
    --no-plot (do not plot the error histogram) 
    --water_level <depth> (set the water level in metres)
    --raw-out <output_file> (set the output raw file name, default is 'terrain.raw')
    --hi-res <hires_file> (set the input hires CSV file, default is 'highres.csv')
    --step <step> (set the step size in metres, default is 1.0)
    --zero (create a blank terrain with all zeroes, combine with raise to create a flat terrain at a specified level)
    --raise-by <N> (raise by N metres, negative raise will lower the terrain
    """
    import argparse

    parser = argparse.ArgumentParser(description="Create SL-style raw file from high-res CSV data.")
    parser.add_argument('-q', '--quiet', action='store_true', help="Run without prompts")
    parser.add_argument('--compare-with', type=str, help="Path to existing raw file for comparison")
    parser.add_argument('--no-plot', action='store_true', help="Do not plot error histogram")
    parser.add_argument('-w', '--water-level', type=float, default=WATER_LEVEL, help="Set water level in metres")
    parser.add_argument('-o', '--raw-out', type=str, default=RAW_OUTFILE, help="Output raw file name")
    parser.add_argument('-i', '--hi-res', type=str, default=CSV_FILE, help="Input hires CSV file")
    parser.add_argument('-s', '--step', type=float, default=STEP, help="Step size in metres, controls whether data will be averaged or not")
    parser.add_argument('-z', '--zero', action='store_true', help="Create a blank terrain with all zeroes, combine with raise-by to create a flat terrain at a specified level.")
    parser.add_argument('-r', '--raise-by', type=float, help="raise by N metres, negative raise will lower the terrain. bounds checking will be applied.")
    parser.add_argument('--ramp', type=float, help="Create a ramp terrain from 0 to N metres.")
    parser.add_argument('--quanta', type=int, help="Quantize heights to 1/N metre steps (e.g. 64 for 1/64m). If set, uses quantized encoding.")

    # Set global variables based on arguments
    global args, region_name, resolution
    args = parser.parse_args()
    
    # how do we check if --depth was provided as an argument

    if not args.quiet and args.water_level is None:
        wl = input(f"Water level in metres [default {WATER_LEVEL}]: ").strip()
        try:
            water_level = float(wl) if wl else WATER_LEVEL
        except ValueError:
            print("Invalid water level; using default.", file=sys.stderr)
            args.water_level = water_level
    if args.zero or args.ramp is not None:
        print_if_not_quiet("Note: a generator (--zero, --ramp==N) specified, input data for hi-res iss ignored")
    else:
        print_if_not_quiet(f"Using CSV file: {args.hi_res}")
    print_if_not_quiet(f"Output raw file: {args.raw_out}")
    print_if_not_quiet(f"Step size: {args.step} metres")
    print_if_not_quiet(f"Water level: {args.water_level} metres")

    if not args.zero and args.ramp is None:
        region_name, resolution = read_csv_header(args.hi_res)
        print_if_not_quiet(f"Region name: {region_name}, Resolution: {resolution}")
        FACTOR = int(args.step / resolution) # thus resoltuion 0.25 and step one gives factor 4

        # load and normalise the data
        triples = load_hires_csv(args.hi_res)
        triples = normalise_coordinates(triples)

        # Build the high‑res grid
        Z_hi = build_high_res_grid(triples, resolution)
        # Report missing values
        if report_missing(Z_hi):
            sys.exit(1)

        # Downsample the grid (not dimension check is aginst the integer coord space not the index)
        Z_hi_trimmed = check_dimensions_and_trim(Z_hi, target_dim=int(TARGET_DIM/resolution))
        Z_ds = downsample_grid_to_step_size(Z_hi_trimmed, factor=FACTOR)
    elif args.ramp is not None:
        ramp_height = args.ramp
        print_if_not_quiet(f"Creating ramp terrain with height {ramp_height} metres upwards")
        Z_ds = create_ramp_terrain(max_height=ramp_height, target_dim=TARGET_DIM)
    else:
        print_if_not_quiet("Creating blank terrain with all zeroes")
        Z_ds = create_blank_terrain(target_dim=TARGET_DIM)

    if args.raise_by is not None:
        raise_by = args.raise_by
        if raise_by < 0:
            print_if_not_quiet(f"Lowering terrain by {abs(raise_by)} metres")
        else:
            print_if_not_quiet(f"Raising terrain by {raise_by} metres")
        Z_ds += raise_by
        # bounds checking
        Z_ds = np.clip(Z_ds, 0, 255*255/128)  # max height in SL is 255*255/128

    # the maps are 0,0 lower left, so we need to flip the grid vertically
    Z_ds = Z_ds[::-1, :]
    
    water_byte = np.uint8(np.clip(int(round(args.water_level)), 0, 255))

    if args.quanta is not None:
        print_if_not_quiet(f"Encoding heights with quantization: 1/{args.quanta} metre steps")
        vec = np.vectorize(lambda h: encode_height_quantized(h, quanta_per_m=args.quanta), otypes=[np.uint8, np.uint8])
    else:
        vec = np.vectorize(encode_height, otypes=[np.uint8, np.uint8])

    # vec = np.vectorize(encode_height, otypes=[np.uint8, np.uint8])
    height_chan, height_unit_chan = vec(Z_ds)

    out_arr = build_terrain_array(
        height_chan,
        height_unit_chan,
        water_byte,
        TARGET_DIM
    )

    write_output_raw(out_arr, target_dim=TARGET_DIM)
    if not args.quiet:
        if not args.no_plot:
            print("Plotting error histogram...")
            compare_and_plot_error(args.raw_out, Z_ds, target_dim=TARGET_DIM)
        else:
            compare_raw_to_hires(
                raw_path=args.raw_out,
                hires_grid=Z_ds,
                target_dim=TARGET_DIM
            )
    if args.compare_with is not None and not args.quiet:
        if not args.no_plot:
            print("Plotting error histogram...")
            compare_and_plot_error(args.compare_with, Z_ds, target_dim=TARGET_DIM)
        else:
            compare_raw_to_hires(
                raw_path=args.compare_with,
                hires_grid=Z_ds,
                target_dim=TARGET_DIM
            )

# Create a main function to encapsulate the script logic
if __name__ == "__main__":
    main()
