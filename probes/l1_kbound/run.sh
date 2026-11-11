#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Runs kbound configurations; waits while a process this script did not start is on GPU 0 or 1.
cd "$(dirname "$0")" || exit
busy() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -E "GPU-1ced79bd|GPU-7910a120" ; }
for cfg in "$@"; do
  while [ -n "$(busy)" ]; do echo "waiting: $(busy)" >&2; sleep 5; done
  # shellcheck disable=SC2086  # a config is several key=value arguments
  timeout 300 ./kbound $cfg
done
