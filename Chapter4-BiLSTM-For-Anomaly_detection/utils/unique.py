#!/usr/bin/python3

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Output unique.py lines')
    parser.add_argument('file', nargs='*', help='Specify input files')
    parser.add_argument('-r', '--reverse', action='store_true', help='Output only duplicated lines')
    parser.add_argument('-i', '--ignore-case', action='store_true', help='Case insenstive, e.g., A=a')
    parser.add_argument('-c', '--count', action='store_true', help='Report countings')
    parser.add_argument('-w', '--words', action='store_true', help='Counting words instead of lines')
    parser.add_argument('-s', '--sort', action='store_true', help='Sort countings')
    parser.add_argument('-t', '--threshold', type=int, default=1, help='Output lines that occur more than T times')
    args = parser.parse_args()

    threshold = 2 if args.reverse else args.threshold

    seen = {}

    myInput = [sys.stdin] if len(args.file) == 0 else args.file

    for fin in myInput:
        if isinstance(fin, str):
            fin = open(fin, 'r')
        for line in fin:
            if args.ignore_case:
                line = line.lower()
            line = [word + '\n' for word in line.split()] if args.words else [line]
            for word in line:
                if word not in seen: seen[word] = 0
                seen[word] += 1
                if not args.count and seen[word] == threshold:
                    sys.stdout.write(word)
    if args.count:
        if args.sort:
            keylist = sorted(seen.keys(), key=lambda k: seen[k], reverse=True)
        else:
            keylist = sorted(seen.keys())
        for k in keylist:
            if seen[k] >= args.threshold:
                sys.stdout.write('\t'.join([str(seen[k]), k]))


if __name__ == '__main__':
    main()
