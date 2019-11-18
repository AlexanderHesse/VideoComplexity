#!/usr/bin/env python
# -*- encoding: utf8 -*-

from __future__ import division

from math import exp, log
import sys

from video_complexity import read_csv, write_csv


def group_by(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(row.get(field), list()).append(row)
    for i in sorted(groups.keys()):
        yield groups[i]

def columns(rows):
    if len(rows) > 0:
        for c in range(len(rows[0])):
            yield rows[0].columns[c], [row[c] for row in rows]


def scaled(data, factors, verbose):
    if verbose:
        for method, factor in factors.iteritems():
            print >> sys.stderr, method, factor
    
    for row in data:
        yield row.map(lambda col, val: val * factors[col] if col in factors and val != None else val)


def geometric_mean(l):
    return exp(sum(log(i) for i in l)/len(l))

def by_geometric_mean(reference_data, apply_data, verbose = False):
    methods = [i for i in reference_data[0].columns if i not in ('path', 'frame_index', 'resolution')]
    data = [row for row in reference_data if None not in [row.get(i) for i in methods]]
        
    by_video = group_by(data, 'path')
    avg_by_video = [[geometric_mean(l) for col, l in columns(rows) if col in methods] for rows in by_video]
    avg_all = [geometric_mean(l) for l in zip(*avg_by_video)]
    avg_all = [i/geometric_mean(avg_all) for i in avg_all]
    
    factors = { m:1/i for m, i in zip(methods, avg_all) }
    
    for row in scaled(apply_data, factors, verbose):
        yield row


def by_first_frame(base, data, verbose = False):
    methods = [i for i in data[0].columns if i not in ('path', 'frame_index', 'resolution')]
    
    by_path = group_by(data, 'path')
    for rows in by_path:
        if rows[0].get('frame_index') == None:
            # it's an image
            for row in rows:
                yield row
            continue
            
        rows = sorted(rows, key = lambda row: row.get('frame_index'))
        first = [row for row in rows if None not in row][0]
        
        factors = { m:base/first.get(m) for m in methods }
        
        for row in scaled(rows, factors, verbose):
            yield row

def by_average(reference_data, base_value, data, verbose):
    avg = lambda l: sum(l) / len(l)
    
    methods = [i for i in reference_data[0].columns if i not in ('path', 'frame_index', 'resolution')]
    reference_data = [row for row in reference_data if None not in [row.get(i) for i in methods]]
    
    by_path = group_by(reference_data, 'path')
    avg_by_path = [[avg(l) for col, l in columns(rows) if col in methods] for rows in by_path]
    avg_all = [avg(l) for l in zip(*avg_by_path)]
    
    factors = { m:base_value/i for m, i in zip(methods, avg_all) }
    
    for row in scaled(data, factors, verbose):
        yield row


def per_pixel(data):
    methods = [i for i in data[0].columns if i not in ('path', 'frame_index', 'resolution')]
    
    for row in data:
        width, height = [int(d) for d in row.get("resolution").split('x')]
        yield row.map(lambda col, val: val / (width * height) if col in methods and val != None else val)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_CSV", nargs='?', help="the csv file to rescale, stdin if not specified")
    parser.add_argument("--per-pixel", help="calculate the average complexity per pixel instead of the whole frame", action="store_true")
    parser.add_argument("--geometric-average", help="scale INPUT_CSV by the geometric average of REFERENCE_FILE", action="store_true")
    parser.add_argument("--first-frame",  help="scale INPUT_CSV by normalizing the first frame of each video to BASE_VALUE", action="store_true")
    parser.add_argument("--average",  help="scale INPUT_CSV by the factors obtained from setting the average frame complexity of each video in REFERENCE_FILE to BASE_VALUE", action="store_true")
    parser.add_argument("--base-value", metavar="BASE_VALUE", type=float, default=1., help="the base value methods will normalize to")
    parser.add_argument("--reference-file", metavar="REFERENCE_FILE", help="reference file for scale methods, INPUT_CSV if not specified")
    parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
    parser.add_argument("-o", "--output", help="write output to file")
    
    args = parser.parse_args()
    
    if args.INPUT_CSV != None:
        data = list(read_csv(open(args.INPUT_CSV)))
    else:
        data = list(read_csv(sys.stdin))
    
    reference_data = data
    if args.reference_file != None:
        reference_data = list(read_csv(open(args.reference_file)))
    
    if args.per_pixel:
        data = list(per_pixel(data))
        reference_data = list(per_pixel(reference_data))
    
    gen = None
    if args.first_frame:
        gen = by_first_frame(args.base_value, data, args.verbose)
    elif args.geometric_average:
        gen = by_geometric_mean(reference_data, data, args.verbose)
    elif args.average:
        gen = by_average(reference_data, args.base_value, data, args.verbose)
    
    if gen == None:
        sys.exit(0)
    
    if args.output != None:
        with open(args.output, 'w') as f:
            write_csv(gen, f)
    else:
        write_csv(gen, sys.stdout)
