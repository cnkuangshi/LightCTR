#!/bin/sh
if [ $# -lt 3 ]; then
    echo "usage: $0 [ps_num] [worker_num] [master_ip_port like 127.0.0.1:17832]"
    exit -1;
fi

export LightCTR_PS_NUM=$1
shift
export LightCTR_WORKER_NUM=$1
shift
export LightCTR_MASTER_ADDR=$1

make master &
make ps &
make worker &

wait
echo
echo
echo "[Build Success]"
echo "Please copy different BIN file to corresponding machine, run Master first"
echo
echo "[or] Press any key to run clunster on standalone mode"
read -n 1

./LightCTR_BIN_Master &

for ((i=0; i<${LightCTR_PS_NUM}; ++i)); do
	./LightCTR_BIN_PS &
done

for ((i=0; i<${LightCTR_WORKER_NUM}; ++i)); do
	./LightCTR_BIN_Worker &
done

wait
