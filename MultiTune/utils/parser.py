import pdb
import re
import statistics
import sys
import os
import numpy as np
import configparser
from sqlparse.sql import Where, Comparison, Identifier, Parenthesis

from shutil import copyfile


class DictParser(configparser.ConfigParser):
    def get_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


class ConfigParser(object):
    def __init__(self, cnf):
        f = open(cnf)
        self._cnf = cnf
        self._knobs = {}
        for line in f:
            if line.strip().startswith('skip-external-locking') \
                    or line.strip().startswith('[') \
                    or line.strip().startswith('#') \
                    or line.strip() == '':
                continue
            try:
                k, _, v = line.strip().split()
                self._knobs[k] = v
            except:
                continue
        f.close()

    def replace(self, tmp='/tmp/tmp.cnf'):
        record_list = []
        f1 = open(self._cnf)
        f2 = open(tmp, 'w')
        for line in f1:
            tpl = line.strip().split()
            if len(tpl) < 1:
                f2.write(line)
            elif tpl[0] in self._knobs:
                record_list.append(tpl[0])
                tpl[2] = self._knobs[tpl[0]]
                f2.write('%s\t\t%s %s\n' % (tpl[0], tpl[1], tpl[2]))
            else:
                f2.write(line)
        for key in self._knobs.keys():
            if not key in record_list:
                f2.write('%s\t\t%s %s\n' % (key, '=', self._knobs[key]))
        f1.close()
        f2.close()
        copyfile(tmp, self._cnf)

    def set(self, k, v):
        self._knobs[k] = v


def parse_args(config_file='config.ini'):
    cf = DictParser()
    cf.read(config_file, encoding='utf8')
    config_dict = cf.get_dict()
    config_dict['tune']['components'] = eval(config_dict['tune']['components'])
    config_dict['tune']['arms'] = str(list(config_dict['tune']['components'].keys()))
    config_dict['tune']['output_file'] = os.path.join('optimize_history', config_dict['tune']['task_id'] + '.res')
    return config_dict['database'], config_dict['tune']


def parse_benchmark_result(file_path, select_file, timeout=10):
    with open(file_path) as f:
        lines = f.readlines()

    with open(select_file) as f:
        lines_select = f.readlines()
    num_sql = len(lines_select)

    latL = []
    lat_dir = {}
    for line in lines[1:]:
        if line.strip() == '':
            continue
        tmp = line.split('\t')[-1].strip()
        latL.append(float(tmp) / 1000)
        type = line.split('\t')[0].strip()
        lat_dir[type] = float(tmp) / 1000

    for i in range(0, num_sql - len(lines[1:])):
        latL.append(timeout)

    # lat95 = np.percentile(latL, 95)
    lat = np.max(latL)
    lat_mean = np.mean(latL)

    return lat, lat_mean, lat_dir


def parse_sysbench_result(file_path):
    tps, latency, qps = 0, 0, 0
    tpsL, latL, qpsL = [], [], []
    with open(file_path) as f:
        lines = f.read()
    temporal_pattern = re.compile(
        "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)"
        " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
    temporal = temporal_pattern.findall(lines)
    for i in temporal:
        tps += float(i[0])
        # latency += float(i[5])
        # qps += float(i[1])
        tpsL.append(float(i[0]))
        # latL.append(float(i[5]))
        # qpsL.append(float(i[1]))
    num_samples = len(temporal)

    tps /= num_samples
    # qps /= num_samples
    # latency /= num_samples
    tps_var = statistics.variance(tpsL)
    # lat_var = statistics.variance(latL)
    # qps_var = statistics.variance(qpsL)
    return -tps, tps_var


def parse_tpcc_result(file_path):
    with open(file_path,'rU') as f:
        lines = f.read()
    tpmC_temporal_pattern = re.compile("Measured tpmC \(NewOrders\) = (\d+.\d+)")
    tpmC_temporal = tpmC_temporal_pattern.findall(lines)
    tpmC = float(tpmC_temporal[0])
    return -tpmC, -1


def strip_config(config):
    config_stripped = {}
    for k in config.keys():
        if '.' in k and k.split('.')[0] in ['knob1', 'knob2']:
            i = k.index('.')
            k_new = k[i + 1:]
            config_stripped[k_new] = config[k]
        else:
            config_stripped[k] = config[k]
    return config_stripped



