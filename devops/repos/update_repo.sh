#!/bin/bash

set -e

HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-../../install/scripts/}
HIPSYCL_PKG_SCRIPT_DIR_ABS=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_SCRIPT_DIR
HIPSYCL_PKG_REPO_BASE_DIR=${HIPSYCL_PKG_REPO_BASE_DIR:-/data/repos/}

HIPSYCL_PKG_BUILD_HIPSYCL=${HIPSYCL_PKG_BUILD_HIPSYCL:-ON}
HIPSYCL_PKG_BUILD_CUDA=${HIPSYCL_PKG_BUILD_CUDA:-OFF}
HIPSYCL_PKG_BUILD_ROCM=${HIPSYCL_PKG_BUILD_ROCM:-OFF}
HIPSYCL_PKG_BUILD_BASE=${HIPSYCL_PKG_BUILD_BASE:-OFF}

HIPSYCL_PKG_NO_BUILD=${HIPSYCL_PKG_NO_BUILD:-OFF}
HIPSYCL_PKG_PACKAGE=${HIPSYCL_PKG_PACKAGE:-ON}
HIPSYCL_PKG_DEPLOY=${HIPSYCL_PKG_DEPLOY:-ON}
HIPSYCL_PKG_TEST=${HIPSYCL_PKG_TEST:-ON}
HIPSYCL_PKG_TEST_PKG=${HIPSYCL_PKG_TEST_PKG:-OFF}

HIPSYCL_REPO_USER=${HIPSYCL_REPO_USER:-illuhad}
HIPSYCL_REPO_BRANCH=${HIPSYCL_REPO_BRANCH:-stable}

HIPSYCL_WITH_CUDA=${HIPSYCL_WITH_CUDA:-ON}
HIPSYCL_WITH_ROCM=${HIPSYCL_WITH_ROCM:-ON}

HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-9}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-1}
HIPSYCL_PKG_LLVM_REPO_BRANCH=${HIPSYCL_PKG_LLVM_REPO_BRANCH:-release/${HIPSYCL_PKG_LLVM_VERSION_MAJOR}.x}

HIPSYCL_PKG_AOMP_RELEASE=${HIPSYCL_PKG_AOMP_RELEASE:-0.7-7}
HIPSYCL_PKG_AOMP_TAG=${HIPSYCL_PKG_AOMP_TAG:-rel_${HIPSYCL_PKG_AOMP_RELEASE}}

HIPSYCL_PKG_CONTAINER_DIR_SUFFIX=${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX:-containers}
HIPSYCL_PKG_CONTAINER_DIR_SUFFIX=${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX}${HIPSYCL_PKG_NAME_SUFFIX}
HIPSYCL_PKG_CONTAINER_DIR_NAME=${HIPSYCL_PKG_LLVM_REPO_BRANCH/release\//llvm-}-aomp-${HIPSYCL_PKG_AOMP_RELEASE}
HIPSYCL_PKG_CONTAINER_DIR=${HIPSYCL_PKG_CONTAINER_DIR:-$HIPSYCL_PKG_SCRIPT_DIR_ABS/${HIPSYCL_PKG_CONTAINER_DIR_NAME}-${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX}}

#[ "$HIPSYCL_PKG_LLVM_REPO_BRANCH" = "release/9.x" ] && HIPSYCL_PKG_CONTAINER_DIR=$HIPSYCL_PKG_SCRIPT_DIR_ABS/clang9-$HIPSYCL_PKG_CONTAINER_DIR_SUFFIX

export HIPSYCL_PKG_CONTAINER_DIR
export HIPSYCL_PKG_BUILD_HIPSYCL
export HIPSYCL_PKG_BUILD_CUDA
export HIPSYCL_PKG_BUILD_ROCM
export HIPSYCL_PKG_BUILD_BASE
export HIPSYCL_PKG_NO_BUILD
export HIPSYCL_PKG_LLVM_REPO_BRANCH
export HIPSYCL_PKG_LLVM_VERSION_MAJOR
export HIPSYCL_PKG_LLVM_VERSION_MINOR
export HIPSYCL_PKG_LLVM_VERSION_PATCH
export HIPSYCL_PKG_AOMP_RELEASE
export HIPSYCL_PKG_AOMP_TAG
export HIPSYCL_REPO_USER
export HIPSYCL_REPO_BRANCH
export HIPSYCL_PKG_TYPE
export HIPSYCL_PKG_NAME_SUFFIX
export HIPSYCL_PKG_DEVOPS_DIR
export HIPSYCL_WITH_CUDA
export HIPSYCL_WITH_ROCM
#export SINGULARITY_TMPDIR=/data/sbalint/singularity_tmp/
#export HIPSYCL_GPG_KEY=B2B75080

[ "$HIPSYCL_PKG_BUILD_CUDA" = "ON" ] || [ "$HIPSYCL_PKG_BUILD_ROCM" = "ON" ] || \
[ "$HIPSYCL_PKG_BUILD_BASE" = "ON" ] && [ "$HIPSYCL_PKG_NO_BUILD" = "OFF" ] &&  \
bash $HIPSYCL_PKG_SCRIPT_DIR_ABS/rebuild-base-images.sh

[ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON" ] && [ "$HIPSYCL_PKG_NO_BUILD" = "OFF" ] && \
bash $HIPSYCL_PKG_SCRIPT_DIR_ABS/rebuild-hipsycl-images.sh

[ "$HIPSYCL_PKG_TEST" = "ON" ] &&  bash $HIPSYCL_PKG_DEVOPS_DIR/test_installation.sh

[ "$HIPSYCL_PKG_PACKAGE" = "ON" ] && bash $HIPSYCL_PKG_DEVOPS_DIR/create_pkgs.sh

[ "$HIPSYCL_PKG_DEPLOY" = "ON" ] && bash $HIPSYCL_PKG_DEVOPS_DIR/create_repos.sh

[ "$HIPSYCL_PKG_TEST_PKG" = "ON" ] && bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh
# cleanup
rm -rf /data/sbalint/singularity_tmp/*
