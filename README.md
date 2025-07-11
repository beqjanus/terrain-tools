# Terrain tools for Second Life and OpenSim

These tools are designed to work with terrain data in Second Life and OpenSim, allowing for the creation and manipulation of terrain files.

These tools (just tool really) are primarily the result of a need to export and reimport terrain without running into the quantisation issues exposed by Second Life's own terrain exporter. 

# create_raw_from_csv_hires_csv.py

The only tool in this repo at present. 
This script reads a CSV file containing (X, Y, Z) triples, builds a high-resolution grid,
downsamples it to a lower resolution, and encodes the heights into a 13-channel raw file.
The output is suitable for use in Second Life or OpenSim.
Not currently tested for sizes other than 256x256.

