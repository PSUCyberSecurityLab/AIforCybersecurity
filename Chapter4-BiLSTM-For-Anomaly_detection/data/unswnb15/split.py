#!/usr/bin/python3
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description='UNSW-NB15 data splitter')
    parser.add_argument('files', help='input csv filename', nargs='+')
    parser.add_argument('-d', '--dir', help='destination directory', default='data')
    parser.add_argument('--head', help='head only first N lines, -1 means all of them', type=int, default=-1)
    args = parser.parse_args()

    for f in args.files:
        with open(f, 'rt', encoding='utf-8') as fin:
            csvfin = csv.reader(fin, delimiter=',')
            for lc, line in enumerate(csvfin):
                if lc == args.head:
                    print("> Extract ", args.head, " lines in file ", f)
                    break
                srcip = line[0]
                dstip = line[2]
                srcdirname = os.path.join(args.dir, srcip)
                dstdirname = os.path.join(args.dir, dstip)
                os.makedirs(srcdirname, exist_ok=True)
                os.makedirs(dstdirname, exist_ok=True)
                linestr = ','.join(line) + '\n'
                with open(os.path.join(srcdirname, 'flow_from_this.csv'), 'at') as fout:
                    fout.write(linestr)
                with open(os.path.join(srcdirname, 'flows.csv'), 'at') as fout:
                    fout.write(linestr)
                with open(os.path.join(dstdirname, 'flow_to_this.csv'), 'at') as fout:
                    fout.write(linestr)
                if srcip == dstip:
                    continue
                with open(os.path.join(dstdirname, 'flows.csv'), 'at') as fout:
                    fout.write(linestr)
        print("> File ", f, " has been extracted successfully.")


if __name__ == '__main__':
    print("=====SPLIT UNSW-NB15 BEGINS=====")
    main()
    print("=====SPLIT UNSW-NB15 ENDS=====")
