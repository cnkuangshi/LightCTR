#/bin/bash
set -x -e

git clone git://github.com/zeromq/libzmq.git || true
cd libzmq
./autogen.sh
./configure --prefix=`pwd`/../zeromq
make && make install