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
    data = [row for row in reference_data if None not in row]
        
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
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="the csv file to rescale")
    parser.add_argument("-g", metavar="REFERENCE_CSV_PATH", help="scale by the geometric average of a reference result csv file")
    parser.add_argument("-b", metavar="BASE_NUM", type=float, help="scale by normalizing to the first frame of each video")
    parser.add_argument("-v", "--verbose", help="verbose", action="store_true")
    parser.add_argument("-o", "--output", help="write output to file")
    
    args = parser.parse_args()
    
    data = list(read_csv(open(args.csv)))
    
    gen = None
    if args.b != None:
        gen = by_first_frame(args.b, data, args.verbose)
    elif args.g != None:
        reference_data = list(read_csv(open(args.g)))
        gen = by_geometric_mean(reference_data, data, args.verbose)
    
    if gen == None:
        sys.exit(0)
    
    if args.output != None:
        with open(args.output, 'w') as f:
            write_csv(gen, f)
    else:
        write_csv(gen, sys.stdout)
