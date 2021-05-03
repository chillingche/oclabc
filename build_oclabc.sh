#!/bin/bash

script_name=$0
build_threads="8"
platform=android-aarch64

export OCLABC_ROOT=$(cd `dirname $0` && pwd)
echo "[INFO] build oclabc in ${OCLABC_ROOT}..."
cd ${OCLABC_ROOT}

options="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-24 \
         -DCMAKE_INSTALL_PREFIX=${OCLABC_ROOT}/install-${platform}"

if [[ ! -d build-${platform} ]]; then
    mkdir build-${platform}
fi
if [[ ! -d install-${platform} ]]; then
    mkdir install-${platform}
fi

cd ${OCLABC_ROOT}/build-${platform}
echo "[INFO] use cmake options ${options}"
cmake .. ${options}
make -j${build_threads} || exit 1
make install || exit 1

cd ${OCLABC_ROOT}

# adb command
echo "[INFO] push files to the phone: "
adb shell "mkdir -v /data/local/tmp/wangzhe_r"
adb push ${OCLABC_ROOT}/install-${platform}/test/* /data/local/tmp/wangzhe_r
adb push ${OCLABC_ROOT}/src/cl/* /data/local/tmp/wangzhe_r