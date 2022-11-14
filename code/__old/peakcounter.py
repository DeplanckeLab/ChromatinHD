#!/bin/env python

import pathlib
import collections

root = pathlib.Path("./")
project_root = root / "output"
laf.set_project_root(project_root)

data_folder = project_root / "data/pbmc10k"
flow = laf.Flow("pmbc10k/full_peak", flow = "datasets")
dataset_folder = flow.path

fragments_path = data_folder / "fragments.tsv"
peaks_path = dataset_folder / "peaks.tsv"

test_run = False
if test_run:
    flow = laf.Flow("test", flow = "datasets")
    data_folder = pathlib.Path("./") / "data" / "test"
    data_folder.mkdir(exist_ok = True)

    fragments_path = data_folder/ "fragments.tsv"
    peaks_path = data_folder/"peaks.tsv"

    fragments_value = """
chr1\t00\t10\ta\t1
chr1\t01\t10\tb\t1
chr1\t05\t20\ta\t1
chr1\t15\t20\tc\t1
chr2\t0\t10\ta\t1
chr2\t0\t10\td\t1
""".strip("\n \t")
    with fragments_path.open("w") as fragments_file:
        fragments_file.write(fragments_value)
        
    peaks_value = """
chr1\t1\t5
chr1\t6\t8
chr1\t6\t15
chr1\t15\t20
chr2\t8\t20
""".strip("\n \t")
    with peaks_path.open("w") as peaks_file:
        peaks_file.write(peaks_value)

peak_queue = collections.deque()

counts = collections.defaultdict(int)

def read_peak(line):
    chromosome, start, end = line.rstrip("\n").split("\t")
    return chromosome, int(start), int(end)


def read_fragment(line):
    chromosome, start, end, barcode, support = line.rstrip("\n").split("\t")
    return chromosome, int(start), int(end), barcode

with fragments_path.open() as fragments_file, peaks_path.open() as peaks_file:
    peak_line = peaks_file.readline()
    while peak_line.startswith("#"):
        peak_line = peaks_file.readline()
    next_peak = read_peak(peaks_file.readline())
    cur_chromosome = next_peak[0]

    peak_ix = 0

    fragment_ix = 0
    # while (fragment_ix < 30000000):
    while (fragment_ix < 300000000):
        # get next fragment
        next_line = fragments_file.readline()

        # EOF check
        if len(next_line) <= 1:
            break
        chromosome, start, end, barcode = read_fragment(next_line)

        # skip peaks if chromosomes do not match
        if cur_chromosome != chromosome:
            peak_queue.clear()
            while next_peak[0] != chromosome:
                next_peak = read_peak(peaks_file.readline())
            cur_chromosome = chromosome
            print(chromosome)

        # remove peaks from the queue that end before the start of the fragment
        while (len(peak_queue) > 0) and (peak_queue[0][1] < start):
            peak_queue.popleft()
        
        # add peaks to the queue that start before the end of the fragment
        if next_peak is not None:
            while (next_peak[0] == cur_chromosome) and (next_peak[1] < end):
                peak_queue.append((next_peak[1], next_peak[2], peak_ix))
                next_line = peaks_file.readline()
                if len(next_line) <= 1:
                    next_peak = None
                    break
                else:
                    next_peak = read_peak(next_line)

                peak_ix += 1

        # add counts
        for peak in peak_queue:
            counts[(barcode, peak[2])] += 1
        # counts[barcode].extend([peak[2] for peak in peak_queue])

        fragment_ix += 1

        if (fragment_ix % 1000000) == 0:
            print(chromosome)

        # check next peak
        if next_peak is None:
            break

flow.counts = counts
