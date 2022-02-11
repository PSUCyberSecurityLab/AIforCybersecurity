#!/usr/bin/python3

import argparse
import csv
import datetime

srcip = 0
sport = 1
dstip = 2
dsport = 3
proto = 4
state = 5
dur = 6
sbytes = 7
dbytes = 8
sttl = 9
dttl = 10
sloss = 11
dloss = 12
service = 13
sload = 14
dload = 15
spkts = 16
dpkts = 17
swin = 18
dwin = 19
stcpb = 20
dtcpb = 21
smeansz = 22
dmeansz = 23
trans_depth = 24
res_bdy_len = 25
sjit = 26
djit = 27
stime = 28
ltime = 29
sintpkt = 30
dintpkt = 31
tcprtt = 32
synack = 33
ackdat = 34
is_sm_ips_ports = 35
ct_state_ttl = 36
ct_flw_http_mthd = 37
is_ftp_login = 38
ct_ftp_cmd = 39
ct_srv_src = 40
ct_srv_dst = 41
ct_dst_ltm = 42
ct_src_ltm = 43
ct_src_dport_ltm = 44
ct_dst_sport_ltm = 45
ct_dst_src_ltm = 46
attack_cat = 47
label = 48


def getDirectionFromLine(line, filepath):
    if line[srcip] == line[dstip]:
        return 'local'
    elif line[srcip] in filepath:
        return 'out'
    elif line[dstip] in filepath:
        return 'in'
    return 'na'


def getKeyFromLine(line, keyset=0, divisor=100):
    # correction
    line[is_ftp_login] = str(int(line[is_ftp_login]) & 1) if len(line[is_ftp_login].strip()) else '0'
    line[ct_ftp_cmd] = str(int(line[ct_ftp_cmd])) if len(line[ct_ftp_cmd].strip()) else '0'
    line[ct_flw_http_mthd] = str(int(line[ct_flw_http_mthd])) if len(line[ct_flw_http_mthd].strip()) else '0'
    line[trans_depth] = str(int(line[trans_depth])) if len(line[trans_depth].strip()) else '0'
    # add-on
    pktsum = str((int(line[spkts]) + int(line[dpkts])) // divisor)
    pktsum_idx = len(line)
    line.append(pktsum)
    # key
    keys = [
        # 706/366 keys when divisor is 100/1000
        [proto, state, service, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, trans_depth, is_sm_ips_ports, pktsum_idx],
        [proto, state, service, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, trans_depth, is_sm_ips_ports],  # 284 keys
        [proto, state, service, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, trans_depth],  # 280 keys
        [proto, state, service, is_ftp_login, ct_ftp_cmd],  # 206 keys
        [proto, state, service, is_ftp_login],  # 182 keys
    ]
    keystr = ','.join([line[k] for k in keys[keyset]])
    print('sbyte:', line[sbytes], ', dbytes:', line[dbytes], end=' , ')
    print('spkts:', line[spkts], ', dpkts:', line[dpkts], end=' , ')
    print('smeansz:', line[smeansz], ', dmeansz:', line[dmeansz])
    print('sum/' + str(divisor) + ':', str(pktsum))
    return keystr


def getLabelFromLine(line):
    if line[label].strip() == '0':
        return None
    else:
        return line[attack_cat].strip()


def getDateTimeFromLine(line, timestamp=stime):
    ts = int(line[timestamp])
    dt = datetime.datetime.fromtimestamp(ts)
    return dt


def main():
    parser = argparse.ArgumentParser(description='Key stats for UNSW-NB15 dataset')
    parser.add_argument('file', help='Files', nargs='+')
    parser.add_argument('-k', '--keyset', type=int, default=0, help="keyset is in (0, 4), different keyset has different processed keys")
    parser.add_argument('--nodi', action='store_true', help='toggle off direction output')
    args = parser.parse_args()

    keystats = {}
    for f in args.file:
        with open(f, 'rt', encoding='utf-8') as fin:
            csvfin = csv.reader(fin, delimiter=',')
            for linecount, line in enumerate(csvfin):
                if args.nodi:
                    keystr = getKeyFromLine(line, args.keyset)
                else:
                    keystr = getDirectionFromLine(line, f) + ',' + getKeyFromLine(line, args.keyset)
                keystats.setdefault(keystr, 0)
                keystats[keystr] += 1

    keysum = sum([keystats[k] for k in keystats])
    keys = sorted(list(keystats.keys()), key=lambda k: keystats[k], reverse=True)
    accumulate = 0
    for i, key in enumerate(keys):
        occurrences = keystats[key]
        accumulate += occurrences
        print('{:5d}'.format(i + 1),
              '{:05.02f}%'.format(accumulate * 100 / keysum),
              '{:05.02f}%'.format(occurrences * 100 / keysum),
              '{:6d}'.format(occurrences), key)


if __name__ == '__main__':
    main()
