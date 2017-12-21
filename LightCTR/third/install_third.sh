#/bin/bash
set -x -e

cd zeromq-4.2.2
./configure --prefix=`pwd`/../zeromq
make && make install