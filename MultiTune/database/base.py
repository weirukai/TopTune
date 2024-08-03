import os
import pdb
import sys
import time
import json
import logging
import threading
import numpy as np
import paramiko
import sqlparse
import threading
import sql_metadata
import subprocess
import gevent
import csv
import multiprocessing as mp
from abc import ABC, abstractmethod
from ConfigSpace import Configuration
from multiprocessing import Manager
from sqlparse.sql import Identifier
from shutil import copyfile
from ..utils.parser import strip_config, parse_tpcc_result

sys.path.append('..')
from ..utils.parser import parse_benchmark_result,parse_sysbench_result


class DB(ABC):
    def __init__(self, task_id, dbtype, host, port, user, passwd, dbname, sock, cnf, knob_config_file_gp, knob_num_gp,knob_config_file_smac, knob_num_smac,
                 workload_name, workload_timeout, workload_qlist_file, workload_qdir, scripts_dir,
                 log_path='./logs', result_path='./results', restart_wait_time=5, **kwargs
                 ):
        # database
        self.task_id = task_id
        self.dbtype = dbtype
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.sock = sock
        self.cnf = cnf
        # self.all_pk_fk = None
        # self.all_columns = None
     
        # logger
        self.log_path = log_path
        self.logger = self.setup_logger()
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        # initialize
        self.iteration = 0

        # workload
        self.workload_name = workload_name.lower()
        self.workload_timeout = float(workload_timeout)
        self.minimum_timeout = float(workload_timeout)
        self.workload_qlist_file = workload_qlist_file
        self.workload_qdir = workload_qdir
        self.scripts_dir = scripts_dir
        self.workload = self.generate_workload()
        self.queries = self.get_queries()
        # remote
        self.remote = eval(kwargs['remote_mode'])
        if self.remote:
            self.ssh_user = kwargs['ssh_user']
            self.ssh_passwd = kwargs['ssh_passwd']
            self.ssh_pk_file = os.path.expanduser('~/.ssh/id_rsa')
            self.pk = paramiko.RSAKey.from_private_key_file(self.ssh_pk_file)
        # knob
        self.mysql8 = eval(kwargs['mysql8'])
        self.knob_num_gp = int(knob_num_gp)
        self.knob_config_file_gp = knob_config_file_gp
        self.knob_details_gp = self.get_knobs(self.knob_num_gp,self.knob_config_file_gp)
        self.knob_num_smac = int(knob_num_smac)
        self.knob_config_file_smac = knob_config_file_smac
        self.knob_details_smac = self.get_knobs(self.knob_num_smac,self.knob_config_file_smac)
        self.restart_wait_time = restart_wait_time
        try:
            self._connect_db()
        except Exception as e:
            self._start_db()

        # internal metrics collect signal
        self.im_alive_init()

    @abstractmethod
    def _connect_db(self):
        pass

    @abstractmethod
    def _execute(self, sql):
        pass

    @abstractmethod
    def _fetch_results(self, sql, json=False):
        pass

    @abstractmethod
    def _close_db(self):
        pass

    @abstractmethod
    def _start_db(self, isolation=False):
        pass

    @abstractmethod
    def _modify_cnf(self, config):
        pass

    @abstractmethod
    def _clear_processlist(self):
        pass

    @abstractmethod
    def get_index_size(self):
        pass


    def apply_knob_config(self, knob_config):
        if len(knob_config.items()) == 0:
            self.logger.debug('No knob changes.')
            return

        try:
            knob_config = strip_config(knob_config)
        except:
            pass

        _knob_config = {}

        # check scale
        for k, v in knob_config.items():
            if k in self.knob_details_gp.keys() and self.knob_details_gp[k.split('.')[1] if '.' in k else k]['type'] == 'integer' and self.knob_details_gp[k.split('.')[1] if '.' in k else k]['max'] > sys.maxsize:
                _knob_config[k] = knob_config[k] * 1000
            else:
                _knob_config[k] = knob_config[k]

        flag = self._modify_cnf(_knob_config)
        if not flag:
            #copyfile(self.cnf.replace('experiment', 'default'), self.cnf)
            raise Exception('Apply knobs failed')
        self.logger.debug("Iteration {}: Knob Configuration Applied to MYCNF!".format(self.iteration))
        self.logger.debug('Knob Config: {}'.format(_knob_config))



    def apply_query_config(self, query_config):
        v = query_config[list(query_config.keys())[0]]
        self.workload_qdir = self.workload_qdir.replace(self.workload_qdir.split('_')[-1], str(v)) + '/'
        self.workload_qlist_file = self.workload_qlist_file.replace(self.workload_qlist_file.split('_')[-1], str(v)) + '.txt'
        self.workload = self.generate_workload(self.workload_qdir, self.workload_qlist_file)


    def apply_config(self,type, config):
        if type == 'knob':
            self.apply_knob_config(config)
        elif type == 'index':
            self.apply_index_config(config)
        elif type == 'query':
            self.apply_query_config(config)
        elif type == 'view':
            self.apply_view_config(config)


    def generate_workload(self, workload_qdir=None, workload_qlist_file=None):
        if workload_qdir is None:
            workload_qdir = self.workload_qdir
        if workload_qlist_file is None:
            workload_qlist_file = self.workload_qlist_file
            
        if self.workload_name == 'job':
            script = os.path.join(self.scripts_dir, 'run_{}.sh'.format(self.dbtype))
            wl = {
                'type': 'read',
                'cmd': 'bash %s %s %s {output} %s %s %s' % (script, workload_qdir, workload_qlist_file, self.passwd, self.dbname, self.workload_timeout * 1000)
            }
        elif self.workload_name == 'sysbench':
            script = os.path.join(self.scripts_dir, 'run_{}_{}.sh'.format(self.dbtype,self.workload_name))
            wl = {
                'type': 'read',
                'cmd': 'bash %s {output}' % (
                script)
            }
        elif self.workload_name == 'tpcc':
            script = os.path.join(self.scripts_dir, 'run_{}_{}.sh'.format(self.dbtype, self.workload_name))
            wl = {
                'type': 'readwrite',
                'cmd': 'bash %s {output}' % (
                    script
                )
            }
        else:
            raise ValueError('Invalid workload name')
        return wl

    def get_queries(self):
        queries = []
        with open(self.workload_qlist_file, 'r') as f:
            query_list = f.read().strip().split('\n').copy()

        for q in query_list:
            qf = os.path.join(self.workload_qdir, q)
            with open(qf, 'r') as f:
                query = f.read().strip()
                queries.append(query)
        return queries


    def generate_benchmark_cmd(self):
        timestamp = int(time.time())
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        filename = self.result_path + '/{}.log'.format(timestamp)
        filename = os.path.join(dirname,filename)

        if self.workload_name == 'job':
            cmd = self.workload['cmd'].format(output=filename)
        elif self.workload_name == 'sysbench':
            cmd = self.workload['cmd'].format(output=filename)
        elif self.workload_name == 'tpcc':
            filename = timestamp
            cmd = self.workload['cmd'].format(output=filename)
        else:
            raise ValueError('Invalid workload name')
        return cmd, filename

    def setup_logger(self):
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        logger = logging.getLogger(self.task_id)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s')
        formatter = logging.Formatter('[{}][%(asctime)s]: %(levelname)s, %(message)s'.format(logger.name))

        p_stream = logging.StreamHandler()
        p_stream.setFormatter(formatter)
        p_stream.setLevel(logging.INFO)

        f_stream = logging.FileHandler(
            os.path.join(self.log_path, '{}.log'.format(self.task_id)), mode='a', encoding='utf8')
        f_stream.setFormatter(formatter)
        f_stream.setLevel(logging.DEBUG)

        logger.addHandler(p_stream)
        logger.addHandler(f_stream)

        return logger


    def evaluate(self, config, collect_im=True):
        #return(np.random.random(), np.random.random()), 0, np.random.random(65)
        self.iteration += 1
        collect_im = False
        if isinstance(config, Configuration):
            config = config.get_dictionary()

        knob_config = {}
        workload_qlist_file = self.workload_qlist_file
        for k, v in config.items():
            knob_config[k[6:]] = v
        self._close_db()
        self.apply_knob_config(knob_config)

        start_success = self._start_db()
        if not start_success:
            raise Exception

        self.logger.debug('restarting mysql, sleeping for {} seconds'.format(self.restart_wait_time))

        # # collect internal metrics
        internal_metrics = Manager().list()
        im = mp.Process(target=self.get_internal_metrics, args=(internal_metrics, self.workload_timeout, 0))
        self.set_im_alive(True)
        if collect_im:
            im.start()

        # run benchmark
        timeout = False
        cmd, filename = self.generate_benchmark_cmd()

        self.logger.debug("Iteration {}: Benchmark start, saving results to {}!".format(self.iteration, filename))
        # self.logger.info(cmd)
        p_benchmark = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
        self.logger.info(cmd)

        try:
            p_benchmark.communicate(timeout=self.workload_timeout)
            p_benchmark.poll()
            self.logger.debug("Iteration {}: Benchmark finished!".format(self.iteration))

        except subprocess.TimeoutExpired:
            timeout = True
            self.logger.debug("Iteration {}: Benchmark timeout!".format(self.iteration))
            self._clear_processlist()

            self.logger.debug("Iteration {}: Clear database processes!".format(self.iteration))

        # stop collecting internal metrics
        im_result = []
        if collect_im:
            self.set_im_alive(False)
            im.join()

            keys = list(internal_metrics[0].keys())
            keys.sort()
            im_result = np.zeros(len(keys))
            for idx in range(len(keys)):
                key = keys[idx]
                data = [x[key] for x in internal_metrics]
                im_result[idx] = float(sum(data)) / len(data)

        # get costs
        time.sleep(1)
        space_cost = self.get_index_size()

        if self.workload_name == 'job':
            dirname, _ = os.path.split(os.path.abspath(__file__))
            time_cost, lat_mean, time_cost_dir = parse_benchmark_result(filename, workload_qlist_file, self.workload_timeout)
            self.time_cost_dir = time_cost_dir
        elif self.workload_name == 'sysbench':
            time_cost, lat_mean = parse_sysbench_result(filename)
        elif self.workload_name == 'tpcc':
            time_cost, lat_mean = parse_tpcc_result(filename)
        else:
            raise ValueError

        self.logger.info("Iteration {}: configuration {}\t time_cost {}\t space_cost {}\t timeout {} \t lat_mean {}".format(
            self.iteration, config, time_cost, space_cost, timeout, lat_mean))

        if time_cost < self.minimum_timeout:
            self.minimum_timeout = time_cost
        return (time_cost, lat_mean), space_cost, im_result

    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value

    def get_internal_metrics(self, internal_metrics, run_time, warmup_time):
        _counter = 0
        _period = 5
        count = (run_time + warmup_time) / _period - 1
        warmup = warmup_time / _period

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(float(_period), collect_metric, (counter,))
            timer.start()
            if counter >= count or not im_alive.value:
                timer.cancel()
            if counter > warmup:
                sql = 'SELECT NAME, COUNT from information_schema.INNODB_METRICS where status="enabled" ORDER BY NAME'
                res = self._fetch_results(sql)
                im_dict = {}
                for (k, v) in res:
                    im_dict[k] = v
                internal_metrics.append(im_dict)

        collect_metric(_counter)
        return internal_metrics

    def get_knobs(self, knob_num, knob_config_file):
        if self.mysql8:
            blacklist = ['query_cache_min_res_unit', 'query_cache_size']
        else:
            blacklist = []

        with open(knob_config_file, 'r') as f:
            knob_tmp = json.load(f)
            knobs = list(knob_tmp.keys())

        i = 0
        count = 0
        knob_details = dict()
        while count < knob_num:
            key = knobs[i]
            if not key in blacklist:
                knob_details[key] = knob_tmp[key]
                count = count + 1

            i = i + 1

        self.logger.info('Initialize {} Knobs'.format(len(knob_details.keys())))
        return knob_details
