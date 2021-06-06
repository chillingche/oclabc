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
ADB_DEVICES=$(adb devices | grep -v List)
ADB_DIR="/data/local/tmp/oclabc"
if [[ ${ADB_DEVICES} == "" ]]; then
    echo "[WARN] found no adb device."
else
    if [[ ${ADB_DEVICES} =~ "device" ]]; then
        ADB_DEVICES=$(echo ${ADB_DEVICES} | head -n 1 | awk {'print $1'})
    fi
    echo "[INFO] push files to the device (${ADB_DEVICES}): "
    adb -s ${ADB_DEVICES} shell "mkdir -v ${ADB_DIR}"
    adb -s ${ADB_DEVICES} push ${OCLABC_ROOT}/install-${platform}/lib/* ${ADB_DIR}
    adb -s ${ADB_DEVICES} push ${OCLABC_ROOT}/install-${platform}/examples/* ${ADB_DIR}
fi
