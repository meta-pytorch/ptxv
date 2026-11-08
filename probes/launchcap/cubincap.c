/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LD_PRELOAD cubin capture through CUPTI: every module the driver loads (including the
// lazily loaded, compressed fatbins of libtorch_cuda that cuobjdump does not list) is
// written to $CUBINCAP_DIR/<n>.cubin, and the handle each module / library load returned to
// $CUBINCAP_DIR/handles.jsonl. Used with launchcap for binary-only PyTorch kernels.
//   gcc -shared -fPIC -O2 -o cubincap.so cubincap.c -I$CUDA/targets/sbsa-linux/include \
//       -L<dir of libcupti.so.13> -lcupti
#define _GNU_SOURCE
#include <cupti.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static __thread int t_file = -1;  // file written by a module load inside the current API call

// $CUBINCAP_DIR/handles.jsonl: {"handle": H, "file": N} for every module / library handle a
// load call returned (decimal, as launchcap's "mod" / "lib"), and for the module of each loaded
// library: specializations of one Triton / CuTeDSL kernel share its name, and only the
// launch's handle tells their images apart (tools/capture.py).
static void log_handle(const char *dir, unsigned long long h, int file) {
    char path[4096];
    snprintf(path, sizeof path, "%s/handles.jsonl", dir);
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "{\"handle\": %llu, \"file\": %d}\n", h, file);
    fclose(f);
}

static const char *out_dir(char *buf, size_t n) {
    const char *dir = getenv("CUBINCAP_DIR");
    if (!dir) dir = ".";
    // "%r" in the directory is the torchrun rank ($RANK; the pid without one), so the
    // processes of a multi-process capture write separate directories.
    const char *at = strstr(dir, "%r");
    if (!at) return dir;
    char rk[32];
    if (getenv("RANK")) snprintf(rk, sizeof rk, "%s", getenv("RANK"));
    else snprintf(rk, sizeof rk, "p%d", (int)getpid());
    snprintf(buf, n, "%.*s%s%s", (int)(at - dir), dir, rk, at + 2);
    return buf;
}

static void CUPTIAPI cb(void *u, CUpti_CallbackDomain d, CUpti_CallbackId id, const void *data) {
    (void)u;
    char dbuf[4096];
    if (d == CUPTI_CB_DOMAIN_DRIVER_API) {
        if (id != CUPTI_DRIVER_TRACE_CBID_cuModuleLoad &&
            id != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData &&
            id != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx &&
            id != CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary &&
            id != CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData &&
            id != CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile)
            return;
        const CUpti_CallbackData *c = (const CUpti_CallbackData *)data;
        if (c->callbackSite == CUPTI_API_ENTER) {
            t_file = -1;
            return;
        }
        // Every one of these takes the output handle pointer first.
        void **hp = c->functionParams ? *(void ***)c->functionParams : 0;
        if (t_file >= 0 && hp && *hp) {
            const char *dir = out_dir(dbuf, sizeof dbuf);
            log_handle(dir, (unsigned long long)*hp, t_file);
            if (id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData ||
                id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile) {
                typedef CUresult (*libmod_t)(CUmodule *, CUlibrary);
                libmod_t lm = (libmod_t)dlsym(RTLD_DEFAULT, "cuLibraryGetModule");
                CUmodule m = 0;
                if (lm && lm(&m, (CUlibrary)*hp) == CUDA_SUCCESS && m)
                    log_handle(dir, (unsigned long long)m, t_file);
            }
        }
        t_file = -1;
        return;
    }
    if (d != CUPTI_CB_DOMAIN_RESOURCE || id != CUPTI_CBID_RESOURCE_MODULE_LOADED) return;
    const CUpti_ResourceData *r = (const CUpti_ResourceData *)data;
    const CUpti_ModuleResourceData *m = (const CUpti_ModuleResourceData *)r->resourceDescriptor;
    if (!m || !m->pCubin || !m->cubinSize) return;
    static int n;
    char path[4096];
    int idx = __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED);
    snprintf(path, sizeof path, "%s/%d.cubin", out_dir(dbuf, sizeof dbuf), idx);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(m->pCubin, 1, m->cubinSize, f);
    fclose(f);
    t_file = idx;
}

__attribute__((constructor)) static void init(void) {
    // A fresh handles.jsonl per process: the .cubin index restarts at 0, so records left by
    // an earlier run in the same directory would name the wrong images.
    char dbuf[4096], path[4096];
    snprintf(path, sizeof path, "%s/handles.jsonl", out_dir(dbuf, sizeof dbuf));
    FILE *f = fopen(path, "w");
    if (f) fclose(f);
    CUpti_SubscriberHandle s;
    if (cuptiSubscribe(&s, (CUpti_CallbackFunc)cb, 0) == CUPTI_SUCCESS) {
        cuptiEnableCallback(1, s, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
        CUpti_CallbackId ids[] = {CUPTI_DRIVER_TRACE_CBID_cuModuleLoad,
                                  CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
                                  CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,
                                  CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
                                  CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,
                                  CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile};
        for (unsigned i = 0; i < sizeof ids / sizeof ids[0]; i++)
            cuptiEnableCallback(1, s, CUPTI_CB_DOMAIN_DRIVER_API, ids[i]);
    }
}
