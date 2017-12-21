#!/bin/sh
if [ $# -lt 3 ]; then
    echo "usage: $0 ps_num worker_num master_ip_port"
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
echo "Build Success."

./LightCTR_BIN_Master &

for ((i=0; i<${LightCTR_PS_NUM}; ++i)); do
	./LightCTR_BIN_PS &
done

for ((i=0; i<${LightCTR_WORKER_NUM}; ++i)); do
	./LightCTR_BIN_Worker &
done

wait
