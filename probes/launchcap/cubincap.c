/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LD_PRELOAD cubin capture through CUPTI: every module the driver loads (including the
// lazily loaded, compressed fatbins of libtorch_cuda that cuobjdump does not list) is
// written to $CUBINCAP_DIR/<n>.cubin. Used with launchcap for binary-only PyTorch kernels.
//   gcc -shared -fPIC -O2 -o cubincap.so cubincap.c -I$CUDA/targets/sbsa-linux/include \
//       -L<dir of libcupti.so.13> -lcupti
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void CUPTIAPI cb(void *u, CUpti_CallbackDomain d, CUpti_CallbackId id, const void *data) {
    (void)u;
    if (d != CUPTI_CB_DOMAIN_RESOURCE || id != CUPTI_CBID_RESOURCE_MODULE_LOADED) return;
    const CUpti_ResourceData *r = (const CUpti_ResourceData *)data;
    const CUpti_ModuleResourceData *m = (const CUpti_ModuleResourceData *)r->resourceDescriptor;
    if (!m || !m->pCubin || !m->cubinSize) return;
    static int n;
    char path[4096];
    const char *dir = getenv("CUBINCAP_DIR");
    if (!dir) dir = ".";
    // "%r" in the directory is the torchrun rank ($RANK; the pid without one), so the
    // processes of a multi-process capture write separate directories.
    const char *at = strstr(dir, "%r");
    char rk[32];
    if (getenv("RANK")) snprintf(rk, sizeof rk, "%s", getenv("RANK"));
    else snprintf(rk, sizeof rk, "p%d", (int)getpid());
    if (at) snprintf(path, sizeof path, "%.*s%s%s/%d.cubin", (int)(at - dir), dir, rk, at + 2, n++);
    else snprintf(path, sizeof path, "%s/%d.cubin", dir, n++);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(m->pCubin, 1, m->cubinSize, f);
    fclose(f);
}

__attribute__((constructor)) static void init(void) {
    CUpti_SubscriberHandle s;
    if (cuptiSubscribe(&s, (CUpti_CallbackFunc)cb, 0) == CUPTI_SUCCESS)
        cuptiEnableCallback(1, s, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
}
