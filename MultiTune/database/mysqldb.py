import os
import pdb
import sys
import threading

import psutil
import pymysql
import time
import mysql.connector
import subprocess
import json
import eventlet
import csv
import multiprocessing as mp

sys.path.append('..')
from ..utils.parser import ConfigParser
from .base import DB
from ..utils.ssh import exec_shell,ssh_init


log_num_default = 2
log_size_default = 50331648


class MysqlDB(DB):
    def __init__(self, *args, mysqld, **kwargs):
        self.mysqld = mysqld
        self.pre_combine_log_file_size = log_num_default * log_size_default

        # internal metrics collect signal
        self.im_alive_init()
        self.sql_dict = {
            'valid': {},
            'invalid': {}
        }

        super().__init__(*args, **kwargs)

    def _connect_db(self):
        conn = pymysql.connect( host= self.host,
                                port= int(self.port),
                                database=self.dbname,
                                charset ="utf8",
                                user= "root",
                                password = self.passwd
                                )
        return conn

    def _execute(self, sql):
        conn = self._connect_db()
        cursor = conn.cursor()
        cursor.execute(sql)
        if cursor: cursor.close()
        if conn: conn.close()

    def _fetch_results(self, sql, json=False):
        conn = self._connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            if cursor: cursor.close()
            if conn: conn.close()

            if json:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            return results
        except:
            return  None

    def _close_db(self):
        mysqladmin = 'mysqladmin'
        # mysqladmin = os.path.dirname(self.mysqld) + '/mysqladmin'
        # kill_cmd = '{} -u{} -S {} shutdown -p{}'.format(mysqladmin, self.user, self.sock, self.passwd)
        kill_cmd = "service mysqld stop"
        force_kill_cmd1 = "ps aux|grep '" + self.sock + "'|awk '{print $2}'|xargs kill -9"
        force_kill_cmd2 = "ps aux|grep '" + self.cnf + "'|awk '{print $2}'|xargs kill -9"
        if self.remote:
            ssh = ssh_init(self)
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(kill_cmd)
            ret_code = ssh_stdout.channel.recv_exit_status()
            if ret_code == 0:
                self.logger.info("Close Mysql successfully")
            else:
                self.logger.info("Force close Mysql!")
                ssh.exec_command(kill_cmd)
            ssh.close()
            self.logger.info('Mysql db is shut down remotely')
        else:
            p_close = subprocess.Popen(kill_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
            try:
                p_close.communicate(timeout=60)
                p_close.poll()
                self.logger.debug('Close MySQL successfully.')
            except subprocess.TimeoutExpired:
                os.system(force_kill_cmd1)
                os.system(force_kill_cmd2)
                self.logger.debug('Force close MySQL!')

    def _start_db(self, isolation=False):
        start_cmd = "service mysqld restart"
        if self.remote:
            ssh = ssh_init(self)
            _, ssh_stdout, _ = ssh.exec_command(start_cmd)
            ret_code = ssh_stdout.channel.recv_exit_status()
            if ret_code != 0:
                self.logger.error("start remote Mysql db error: [{}]".format(ssh_stdout))
            self.pid = 2456
            ssh.close()
        else:
            proc = subprocess.Popen(start_cmd, shell=True, stderr=subprocess.STDOUT,
                                    stdout=subprocess.PIPE,
                                    close_fds=True)
            # self.pid = proc.pid
            try:
                out, err = proc.communicate()
                err_code = proc.poll()
                if err_code:
                    print("failed restart")
                for proc in psutil.process_iter(['name']):
                    if proc.name() == 'mysqld':
                        self.pid = proc.pid
            except Exception as e:
                print(e)
        # proc = subprocess.Popen([self.mysqld, '--defaults-file={}'.format(self.cnf)])
        # self.pid = proc.pid
        if isolation:
            cmd = 'sudo cgclassify -g memory,cpuset:server ' + str(self.pid)
            p = os.system(cmd)
            if not p:
                self.logger.debug('add {} to memory,cpuset:server'.format(self.pid))
            else:
                self.logger.debug('Failed: add {} to memory,cpuset:server'.format(self.pid))

        time.sleep(self.restart_wait_time)
        # test connection
        count = 0
        start_success = True
        while True:
            try:
                conn = self._connect_db()
                if conn.open:
                    self.logger.debug('Start MySQL successfully. PID {}.'.format(self.pid))
                    # self.logger.debug('Wait {} seconds for connection'.format(count))
                    conn.close()
                    self.pre_combine_log_file_size = float(self.get_varialble_value("innodb_log_file_size")) * float(
                        self.get_varialble_value("innodb_log_files_in_group"))
                    break
            except:
                time.sleep(1)
                count = count + 1
                if count > 30:
                    start_success = False
                    self.logger.error("Can not start MySQL!")
                    break

        return start_success

    def _modify_cnf(self, config):
        modify_concurrency = False
        if 'innodb_log_file_size' in config.keys():
            log_size = config['innodb_log_file_size']
        else:
            log_size = log_size_default
        if 'innodb_log_files_in_group' in config.keys():
            log_num = config['innodb_log_files_in_group']
        else:
            log_num = log_num_default
        self.pre_combine_log_file_size = log_size * log_num
        if 'innodb_thread_concurrency' in config.keys() and config['innodb_thread_concurrency'] * (
                200 * 1024) > self.pre_combine_log_file_size:
            true_concurrency = config['innodb_thread_concurrency']
            modify_concurrency = True
            config['innodb_thread_concurrency'] = int(self.pre_combine_log_file_size / (200 * 1024.0)) - 2
            self.logger.info("modify innodb_thread_concurrency")
        if 'innodb_thread_concurrency' in config.keys() and config['innodb_thread_concurrency'] * (
                200 * 1024) > log_num * log_size:
            self.logger.info("innodb_thread_concurrency is set too large")
            config['innodb_thread_concurrency'] = int(self.pre_combine_log_file_size / (200 * 1024.0)) - 2
            # return False

        # os.system(r'\cp /etc/my_backup.cnf /etc/my.cnf')
        cnf = self.cnf
        if self.remote:
            cnf = 'default.cnf'
            try:
                ssh = ssh_init(self)
                sftp = ssh.open_sftp()
                sftp.get(self.cnf, cnf)
            except IOError as e:
                raise Exception(e)
            finally:
                if sftp: sftp.close()
                if ssh: ssh.close()
        cnf_parser = ConfigParser(cnf)
        knobs_not_in_cnf = []
        for key in config.keys():
            if key not in self.knob_details_gp.keys() and key not in self.knob_details_smac.keys():
                knobs_not_in_cnf.append(key)
                continue
            cnf_parser.set(key, config[key])
        cnf_parser.replace(os.path.join(self.log_path, 'tmp.cnf'))

        # 远程需要将本地修改后的cnf文件上传到远端
        if self.remote:
            local_cnf = cnf
            remote_cnf = self.cnf
            ssh = ssh_init(self)
            sftp = ssh.open_sftp()
            try:
                sftp.put(local_cnf, remote_cnf)
            except Exception as e:
                self.logger.error('cnf not exists!')
                raise Exception(e)
            if sftp: sftp.close()
            if ssh: ssh.close()

        self.logger.debug('Modify db config file successfully.')
        if len(knobs_not_in_cnf):
            self.logger.debug('Append knobs: {}'.format(knobs_not_in_cnf))
        return True

    def _clear_processlist(self):
        mysqladmin = os.path.dirname(self.mysqld) + '/mysqladmin'
        clear_cmd = mysqladmin + ' processlist -uroot -S ' + self.sock + """ | awk '$2 ~ /^[0-9]/ {print "KILL "$2";"}' | mysql -uroot -S """ + self.sock
        subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
        res = self._fetch_results("select id from information_schema.processlist where user='root';")
        for id in res:
            id = id[0]
            try:
                self._execute("Kill " + str(id))
            except:
                continue

    def get_index_size(self):
        sql = "SELECT ROUND(SUM(index_length)/(1024*1024), 2) AS 'Total Index Size' FROM INFORMATION_SCHEMA.TABLES  WHERE table_schema='%s';" % self.dbname
        index_size = self._fetch_results(sql, json=False)[0][0]
        return float(index_size)


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

    def im_alive_init(self):
        global im_alive
        im_alive = mp.Value('b', True)

    def set_im_alive(self, value):
        im_alive.value = value











