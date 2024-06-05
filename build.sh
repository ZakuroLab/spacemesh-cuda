#!/bin/bash
HOME_DIR=`pwd`
rm -rf ${HOME_DIR}/build
mkdir ${HOME_DIR}/build
cmake -DCMAKE_BUILD_TYPE=Release -S ${HOME_DIR} -B ${HOME_DIR}/build -DWITH_TEST=ON
cmake --build ${HOME_DIR}/build -j
