cd [path]/oltpbench-master
export OLTPBENCH_HOME=[path]/oltpbench-master
./oltpbenchmark -b tpcc -c config/sample_tpcc_config.xml --execute=true -s 5 -o $1