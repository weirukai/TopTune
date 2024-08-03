cd /root/sysbench
./sysbench oltp_read_write \
        --mysql-host=[hostname] \
        --mysql-port=3306 \
        --mysql-user=root \
        --mysql-password=[password] \
        --mysql-db=sysbench \
        --db-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
        --tables=50 \
        --table-size=800000 \
        --db-ps-mode=disable \
        --report-interval=10 \
        --warmup-time=0 \
        --point-selects=0 \
        --threads=128 \
        --time=180 \
        --db-ps-mode=disable \
        run > $1
