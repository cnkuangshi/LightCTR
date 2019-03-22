#!/bin/sh
if [ $# -lt 2 ]; then
    echo "usage: $0 [worker_num] [master_ip_port like 127.0.0.1:17832]"
    exit -1;
fi

cd ./LightCTR/third
sh ./install_third.sh
cd ../../

export LightCTR_PS_NUM=0
export LightCTR_WORKER_NUM=$1
shift
export LightCTR_MASTER_ADDR=$1

make ring_master &
make ring_worker &

wait
echo
echo
echo "[Build Success]"
echo "Please copy different BIN file to corresponding machine, DON'T forget export LightCTR_WORKER_NUM and LightCTR_MASTER_ADDR, run Master first"
