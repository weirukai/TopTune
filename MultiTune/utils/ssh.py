import subprocess
import logging
import paramiko
# logger = logging.getLogger(__name__)
paramiko_logger = logging.getLogger("paramiko")
paramiko_logger.setLevel(logging.CRITICAL)
def ssh_init(self):
    """
    初始化ssh连接
    :param self: 配置对象
    :return:
    """
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, port=22, username=self.ssh_user, password=self.ssh_passwd, pkey=self.pk,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
        return ssh
    except Exception as e:
        # print(e)
        return None

def exec_shell(shell):
    """
    执行shell命令
    :param shell:
    :return: code,msg
    """
    global process
    try:
        process = subprocess.Popen(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
        output = process.stdout.read()
        code = 0
    except subprocess.CalledProcessError as e:
        code = e.returncode
        output = e.output
        # logger.error('%s execute error: exit_status [%s] err [%s]' % (shell, str(code), output))
    process.kill()
    return code, output.decode('UTF-8')