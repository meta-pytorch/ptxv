/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// In-process launch capture through CUPTI callbacks (ptxv.capture, docs/python.md).
//
// Loaded with ctypes into a running process; no LD_PRELOAD. libcupti and libcuda are resolved
// with dlopen at ptxvcap_init (the libcupti already in the process when there is one), so the
// library links against neither.
//
// - Module images: the resource callback CUPTI_CBID_RESOURCE_MODULE_LOADED writes every cubin
//   the driver loads to <module dir>/<fnv64>.cubin (content-addressed, written once) and keeps
//   the hashes this process loaded (ptxvcap_modules). Enabled from ptxvcap_init on, so modules
//   loaded before a capture region are known.
// - Launches (between ptxvcap_start and ptxvcap_stop, the only time these callbacks are
//   enabled): driver-API callbacks on cuLaunchKernel,
//   cuLaunchKernelEx, cuLaunchCooperativeKernel (and _ptsz) append one JSON line per launch to
//   <dir>/launches.jsonl, in tools/capture.py's launchcap format (name, grid, block, smem,
//   cluster, pdl, stream, params with offsets and hex bytes; "packed" for an `extra` buffer).
// - Snapshots: at API_ENTER the context is synchronized and every device allocation an 8-byte
//   parameter word points into (cuMemGetAddressRange) is copied to <dir>/snap/s<N>_pre_<base>.bin;
//   at API_EXIT, after another synchronize, the same allocations go to s<N>_post_<base>.bin.
//   Synchronizing and copying from a driver callback is legal: the callback runs on the
//   launching thread outside the driver lock, and the calls it makes are not ones the
//   subscriber enabled, so they do not recurse. Launches into a stream under graph capture
//   are recorded without snapshots ("capturing": 1); synchronizing would invalidate the capture.
// - cuGraphLaunch is counted (graph_launches), not recorded: its kernels are not captured.
#define _GNU_SOURCE
#include <cuda.h>
#include <cupti.h>  // includes generated_cuda_meta.h (the launch parameter structs)
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef CUptiResult (*subscribe_t)(CUpti_SubscriberHandle *, CUpti_CallbackFunc, void *);
typedef CUptiResult (*enable_t)(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain,
                                CUpti_CallbackId);
typedef CUptiResult (*unsubscribe_t)(CUpti_SubscriberHandle);
static subscribe_t p_subscribe;
static enable_t p_enable;
static unsubscribe_t p_unsubscribe;

typedef CUresult (*sync_t)(void);
typedef CUresult (*range_t)(CUdeviceptr *, size_t *, CUdeviceptr);
typedef CUresult (*d2h_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*fparam_t)(CUfunction, size_t, size_t *, size_t *);
typedef CUresult (*kparam_t)(CUkernel, size_t, size_t *, size_t *);
typedef CUresult (*fname_t)(const char **, CUfunction);
typedef CUresult (*kname_t)(const char **, CUkernel);
typedef CUresult (*iscap_t)(CUstream, CUstreamCaptureStatus *);
typedef CUresult (*getdev_t)(CUdevice *);
typedef CUresult (*fmod_t)(CUmodule *, CUfunction);
typedef CUresult (*klib_t)(CUlibrary *, CUkernel);
typedef CUresult (*libmod_t)(CUmodule *, CUlibrary);
static sync_t c_sync;
static range_t c_range;
static d2h_t c_d2h;
static fparam_t c_fparam;
static kparam_t c_kparam;
static fname_t c_fname;
static kname_t c_kname;
static iscap_t c_iscap;
static getdev_t c_getdev;
static fmod_t c_fmod;
static klib_t c_klib;
static libmod_t c_libmod;

static CUpti_SubscriberHandle sub;
static int inited;
// Recursive: a lazily loaded module can arrive (on_module) inside a launch being recorded.
static pthread_mutex_t mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// Capture state (guarded by mu).
static int active;
static char cap_dir[4096];
static char mod_dir[4096];
static int snap_mode;  // 0 none, 1 every launch, 2 first launch of each (name, shape)
static unsigned long long max_alloc;   // larger allocations are not snapshotted
static unsigned long long snap_budget; // total snapshot bytes left
static unsigned seq;                   // launches recorded since ptxvcap_start
static unsigned long long graph_launches;
static unsigned long long skipped_snaps;
static char tag[512];

// Module hashes loaded by this process.
static unsigned long long *mods;
static size_t nmods, capmods;

// Per-thread: the allocations snapshotted at API_ENTER of the launch in flight.
#define MAXALLOC 512
static __thread CUdeviceptr t_base[MAXALLOC];
static __thread size_t t_size[MAXALLOC];
static __thread unsigned t_n, t_seq;
static __thread int t_snap;

// First-launch keys (snap_mode 2): FNV of name and shape.
static unsigned long long *keys;
static size_t nkeys, capkeys;

static unsigned long long fnv(const void *p, size_t n, unsigned long long h) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static void write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, 1, n, f);
    fclose(f);
}

// Which image each module / library handle holds: module and library loads record the
// handle at API_EXIT with the image the resource callback saw during the call, so a launch
// names its exact image even when several modules define the same kernel name.
#define MAXHANDLES 65536
static void *h_key[MAXHANDLES];
static unsigned long long h_val[MAXHANDLES];
static size_t nhandles;
static __thread unsigned long long t_call_mod;  // image loaded during the current API call

static void handle_put(void *k, unsigned long long v) {
    if (!k || !v) return;
    for (size_t i = 0; i < nhandles; i++)
        if (h_key[i] == k) {
            h_val[i] = v;
            return;
        }
    if (nhandles < MAXHANDLES) {
        h_key[nhandles] = k;
        h_val[nhandles++] = v;
    }
}

static unsigned long long handle_get(void *k) {
    for (size_t i = 0; i < nhandles; i++)
        if (h_key[i] == k) return h_val[i];
    return 0;
}

static void on_module(const CUpti_ResourceData *r) {
    const CUpti_ModuleResourceData *m = (const CUpti_ModuleResourceData *)r->resourceDescriptor;
    if (!m || !m->pCubin || !m->cubinSize) return;
    unsigned long long h = fnv(m->pCubin, m->cubinSize, 0xcbf29ce484222325ull);
    pthread_mutex_lock(&mu);
    if (mod_dir[0]) {
        char path[4200];
        snprintf(path, sizeof path, "%s/%016llx.cubin", mod_dir, h);
        struct stat st;
        if (stat(path, &st) != 0 || (size_t)st.st_size != m->cubinSize) {
            char tmp[4300];
            snprintf(tmp, sizeof tmp, "%s.%d.tmp", path, (int)getpid());
            write_file(tmp, m->pCubin, m->cubinSize);
            rename(tmp, path);
        }
    }
    if (nmods == capmods) {
        capmods = capmods ? 2 * capmods : 64;
        mods = realloc(mods, capmods * sizeof *mods);
    }
    mods[nmods++] = h;
    t_call_mod = h;
    pthread_mutex_unlock(&mu);
}

// Packed parameter bytes as the kernel sees them, for the pointer scan.
static unsigned char pbuf[1 << 16];
static size_t plen;
static void put_param(size_t off, const void *p, size_t n) {
    if (off + n <= sizeof pbuf) {
        memcpy(pbuf + off, p, n);
        if (off + n > plen) plen = off + n;
    }
}

static void hex(FILE *f, const unsigned char *p, size_t n) {
    static const char *d = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        fputc(d[p[i] >> 4], f);
        fputc(d[p[i] & 15], f);
    }
}

static void json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fputc('\\', f);
        if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", *s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

static void snap(const char *which) {
    char path[4300];
    for (unsigned i = 0; i < t_n; i++) {
        void *h = malloc(t_size[i]);
        if (!h) continue;
        if (c_d2h(h, t_base[i], t_size[i]) == CUDA_SUCCESS) {
            snprintf(path, sizeof path, "%s/snap/s%u_%s_%llx.bin", cap_dir, t_seq, which,
                     (unsigned long long)t_base[i]);
            write_file(path, h, t_size[i]);
        }
        free(h);
    }
}

// Pointer tables larger than this are not scanned for indirect targets.
#define PTR_SCAN_MAX (1u << 20)

// Add the allocation containing device address v to the snapshot targets; returns the bytes
// added (0 if v is not a device address, already a target, too large or past MAXALLOC).
static unsigned long long add_target(unsigned long long v) {
    CUdeviceptr b;
    size_t z;
    if (!v || c_range(&b, &z, (CUdeviceptr)v) != CUDA_SUCCESS) return 0;
    if (z > max_alloc) {
        skipped_snaps++;
        return 0;
    }
    unsigned i = 0;
    while (i < t_n && t_base[i] != b) i++;
    if (i < t_n || t_n == MAXALLOC) return 0;
    t_base[t_n] = b;
    t_size[t_n++] = z;
    return z;
}

// Record one launch (API_ENTER). Called with mu held.
static void on_launch_enter(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                            unsigned by, unsigned bz, unsigned smem, CUstream s,
                            const CUlaunchConfig *cfg, void **params, void **extra,
                            const char *sym, int cbid) {
    unsigned cx = 1, cy = 1, cz = 1;
    // Preferred cluster dimension (0s without the attribute): S-CGA-SHAPE, probes/cga_size.
    unsigned px = 0, py = 0, pz = 0;
    int pdl = 0;
    if (cfg) {
        for (unsigned i = 0; i < cfg->numAttrs; i++) {
            if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
                cx = cfg->attrs[i].value.clusterDim.x;
                cy = cfg->attrs[i].value.clusterDim.y;
                cz = cfg->attrs[i].value.clusterDim.z;
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION) {
                px = cfg->attrs[i].value.preferredClusterDim.x;
                py = cfg->attrs[i].value.preferredClusterDim.y;
                pz = cfg->attrs[i].value.preferredClusterDim.z;
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
                pdl = cfg->attrs[i].value.programmaticStreamSerializationAllowed != 0;
            }
        }
        if (!cx) cx = cy = cz = 1;  // no cluster attribute
    }
    // The runtime may hand a CUkernel where a CUfunction is expected (launchcap.c).
    CUkernel k = 0;
    size_t o0, z0;
    if (f && c_fparam && c_fparam(f, 0, &o0, &z0) != CUDA_SUCCESS && c_kparam &&
        c_kparam((CUkernel)f, 0, &o0, &z0) == CUDA_SUCCESS)
        k = (CUkernel)f;
    const char *nm = sym;
    if (k && c_kname) c_kname(&nm, k);
    else if (f && c_fname && c_fname(&nm, f) != CUDA_SUCCESS) nm = sym;
    if (!nm) nm = "?";
    char path[4200];
    snprintf(path, sizeof path, "%s/launches.jsonl", cap_dir);
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
    if (c_iscap) c_iscap(s, &cs);
    CUdevice dev = -1;
    if (c_getdev) c_getdev(&dev);
    t_seq = seq++;
    fprintf(fp, "{\"seq\": %u, \"name\": ", t_seq);
    json_str(fp, nm);
    fprintf(fp, ", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], \"smem\": %u, "
                "\"cluster\": [%u, %u, %u], \"pdl\": %d, \"stream\": %llu, \"dev\": %d, "
                "\"cbid\": %d, \"capturing\": %d, \"tag\": ",
            gx, gy, gz, bx, by, bz, smem, cx, cy, cz, pdl, (unsigned long long)s, (int)dev, cbid,
            cs != CU_STREAM_CAPTURE_STATUS_NONE);
    json_str(fp, tag);
    {
        // The launched kernel's module image (0: unknown).
        unsigned long long img = 0;
        CUlibrary lib = 0;
        CUmodule mod = 0;
        if (k && c_klib && c_klib(&lib, k) == CUDA_SUCCESS) img = handle_get(lib);
        if (!img && f && !k && c_fmod && c_fmod(&mod, f) == CUDA_SUCCESS) img = handle_get(mod);
        if (!img && t_call_mod) img = t_call_mod;  // loaded lazily by this very launch
        if (img) fprintf(fp, ", \"module\": \"%016llx\"", img);
    }
    if (px) fprintf(fp, ", \"preferred_cluster\": [%u, %u, %u]", px, py, pz);
    {
        // Wall-clock launch time (us), the clock of torch's allocator trace (time_us): the
        // case writer keeps only allocator blocks live at the launch.
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        fprintf(fp, ", \"t_us\": %llu",
                (unsigned long long)ts.tv_sec * 1000000ull + (unsigned long long)ts.tv_nsec / 1000ull);
    }
    fprintf(fp, ", \"params\": [");
    plen = 0;
    if (!params && extra) {
        const unsigned char *buf = 0;
        size_t bsize = 0;
        for (void **e = extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
            if (e[0] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = (const unsigned char *)e[1];
            if (e[0] == CU_LAUNCH_PARAM_BUFFER_SIZE) bsize = *(size_t *)e[1];
        }
        if (buf) {
            fprintf(fp, "{\"offset\": 0, \"packed\": true, \"hex\": \"");
            hex(fp, buf, bsize);
            put_param(0, buf, bsize);
            fprintf(fp, "\"}");
        }
    }
    for (size_t i = 0; params; i++) {
        size_t off, size;
        CUresult r = k ? c_kparam(k, i, &off, &size)
                       : (c_fparam ? c_fparam(f, i, &off, &size) : CUDA_ERROR_NOT_SUPPORTED);
        if (r != CUDA_SUCCESS) break;
        fprintf(fp, "%s{\"offset\": %zu, \"hex\": \"", i ? ", " : "", off);
        hex(fp, (const unsigned char *)params[i], size);
        put_param(off, params[i], size);
        fprintf(fp, "\"}");
    }
    fprintf(fp, "]");
    // Snapshot decision.
    t_snap = snap_mode && cs == CU_STREAM_CAPTURE_STATUS_NONE;
    if (t_snap && snap_mode == 2) {
        unsigned dims[9] = {gx, gy, gz, bx, by, bz, smem, cx, cy};
        unsigned long long key = fnv(dims, sizeof dims, fnv(nm, strlen(nm), 0xcbf29ce484222325ull));
        for (size_t i = 0; i < nkeys && t_snap; i++)
            if (keys[i] == key) t_snap = 0;
        if (t_snap) {
            if (nkeys == capkeys) {
                capkeys = capkeys ? 2 * capkeys : 64;
                keys = realloc(keys, capkeys * sizeof *keys);
            }
            keys[nkeys++] = key;
        }
    }
    t_n = 0;
    if (t_snap) {
        unsigned long long total = 0;
        // Parameter words that point into device memory (first level of indirection).
        unsigned long long pv[256];
        unsigned npv = 0;
        for (size_t o = 0; o + 8 <= plen; o += 8) {
            unsigned long long v;
            memcpy(&v, pbuf + o, 8);
            unsigned long long z = add_target(v);
            total += z;
            if (v >= 0x10000 && !(v >> 52) && npv < 256) pv[npv++] = v;
        }
        // One level of indirection: pointer tables (an array of tensor addresses that the
        // kernel loads, e.g. multi-tensor optimizers) a parameter points to. The window from
        // the pointer up to PTR_SCAN_MAX bytes (not the whole driver allocation: torch's small
        // tensors share 2 MiB segments) is scanned; the targets are snapshotted too and the case
        // writer turns the words into `ptrs`.
        if (npv) c_sync();
        unsigned long long *h = npv ? malloc(PTR_SCAN_MAX) : 0;
        for (unsigned i = 0; h && i < npv; i++) {
            CUdeviceptr b;
            size_t z;
            if (c_range(&b, &z, (CUdeviceptr)pv[i]) != CUDA_SUCCESS) continue;
            size_t avail = (size_t)(b + z - pv[i]);
            size_t n = avail < PTR_SCAN_MAX ? avail : PTR_SCAN_MAX;
            n &= ~(size_t)7;
            if (!n || c_d2h(h, (CUdeviceptr)pv[i], n) != CUDA_SUCCESS) continue;
            unsigned long long last = 0;
            for (size_t w = 0; w < n / 8; w++) {
                unsigned long long v = h[w];
                if (v < 0x10000 || v >> 52 || v == last) continue;
                last = v;
                total += add_target(v);
            }
        }
        free(h);
        if (2 * total > snap_budget) {
            t_snap = 0;
            t_n = 0;
            skipped_snaps++;
        } else {
            snap_budget -= 2 * total;
        }
    }
    if (t_snap) fprintf(fp, ", \"snap\": %u", t_seq);
    fprintf(fp, "}\n");
    fclose(fp);
    if (t_snap) {
        c_sync();
        snap("pre");
    }
}

static void CUPTIAPI cb(void *u, CUpti_CallbackDomain d, CUpti_CallbackId id, const void *data) {
    (void)u;
    if (d == CUPTI_CB_DOMAIN_RESOURCE) {
        if (id == CUPTI_CBID_RESOURCE_MODULE_LOADED) on_module((const CUpti_ResourceData *)data);
        return;
    }
    if (d != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData *c = (const CUpti_CallbackData *)data;
    if (id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoad || id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData ||
        id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx ||
        id == CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData ||
        id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile) {
        if (c->callbackSite == CUPTI_API_ENTER) {
            t_call_mod = 0;
        } else if (t_call_mod && c->functionParams) {
            // Every one of these takes the output handle pointer first.
            void **hp = *(void ***)c->functionParams;
            if (hp && *hp) {
                pthread_mutex_lock(&mu);
                handle_put(*hp, t_call_mod);
                pthread_mutex_unlock(&mu);
                if ((id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData ||
                     id == CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile) && c_libmod) {
                    CUmodule m = 0;
                    if (c_libmod(&m, (CUlibrary)*hp) == CUDA_SUCCESS && m) {
                        pthread_mutex_lock(&mu);
                        handle_put(m, t_call_mod);
                        pthread_mutex_unlock(&mu);
                    }
                }
            }
            t_call_mod = 0;
        }
        return;
    }
    if (id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
        if (c->callbackSite == CUPTI_API_ENTER) {
            pthread_mutex_lock(&mu);
            if (active) graph_launches++;
            pthread_mutex_unlock(&mu);
        }
        return;
    }
    if (c->callbackSite == CUPTI_API_ENTER) {
        t_call_mod = 0;
        pthread_mutex_lock(&mu);
        if (!active) {
            pthread_mutex_unlock(&mu);
            t_snap = 0;
            return;
        }
        switch (id) {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: {
            const cuLaunchKernel_params *p = c->functionParams;
            on_launch_enter(p->f, p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX,
                            p->blockDimY, p->blockDimZ, p->sharedMemBytes, p->hStream, 0,
                            p->kernelParams, p->extra, c->symbolName, (int)id);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz: {
            const cuLaunchCooperativeKernel_params *p = c->functionParams;
            on_launch_enter(p->f, p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX,
                            p->blockDimY, p->blockDimZ, p->sharedMemBytes, p->hStream, 0,
                            p->kernelParams, 0, c->symbolName, (int)id);
            break;
        }
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
            const cuLaunchKernelEx_params *p = c->functionParams;
            const CUlaunchConfig *g = p->config;
            on_launch_enter(p->f, g->gridDimX, g->gridDimY, g->gridDimZ, g->blockDimX,
                            g->blockDimY, g->blockDimZ, g->sharedMemBytes, g->hStream, g,
                            p->kernelParams, p->extra, c->symbolName, (int)id);
            break;
        }
        default:
            t_snap = 0;
        }
        // mu stays held until API_EXIT when snapshotting, so launches from other threads
        // cannot interleave between the two snapshots.
        if (!t_snap) pthread_mutex_unlock(&mu);
    } else if (t_snap) {
        c_sync();
        snap("post");
        t_snap = 0;
        pthread_mutex_unlock(&mu);
    }
}

// Enabled only while capturing (ptxvcap_start .. ptxvcap_stop): outside a capture region the
// process's launches do not enter this library at all.
static const CUpti_CallbackId launch_ids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,
    CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
};

// Returns 0, or: -1 libcupti not loadable, -2 libcuda not loadable, or a CUptiResult from
// cuptiSubscribe (39 = CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED: a profiler holds it).
int ptxvcap_init(const char *cupti_path, const char *module_dir) {
    pthread_mutex_lock(&mu);
    if (module_dir) snprintf(mod_dir, sizeof mod_dir, "%s", module_dir);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return 0;
    }
    void *h = dlopen(cupti_path, RTLD_NOW | RTLD_GLOBAL);
    void *cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    pthread_mutex_unlock(&mu);
    if (!h) return -1;
    if (!cu) return -2;
    p_subscribe = (subscribe_t)dlsym(h, "cuptiSubscribe");
    p_enable = (enable_t)dlsym(h, "cuptiEnableCallback");
    p_unsubscribe = (unsubscribe_t)dlsym(h, "cuptiUnsubscribe");
    c_sync = (sync_t)dlsym(cu, "cuCtxSynchronize");
    c_range = (range_t)dlsym(cu, "cuMemGetAddressRange_v2");
    c_d2h = (d2h_t)dlsym(cu, "cuMemcpyDtoH_v2");
    c_fparam = (fparam_t)dlsym(cu, "cuFuncGetParamInfo");
    c_kparam = (kparam_t)dlsym(cu, "cuKernelGetParamInfo");
    c_fname = (fname_t)dlsym(cu, "cuFuncGetName");
    c_kname = (kname_t)dlsym(cu, "cuKernelGetName");
    c_iscap = (iscap_t)dlsym(cu, "cuStreamIsCapturing");
    c_getdev = (getdev_t)dlsym(cu, "cuCtxGetDevice");
    c_fmod = (fmod_t)dlsym(cu, "cuFuncGetModule");
    c_klib = (klib_t)dlsym(cu, "cuKernelGetLibrary");
    c_libmod = (libmod_t)dlsym(cu, "cuLibraryGetModule");
    if (!p_subscribe || !p_enable || !c_sync || !c_range || !c_d2h) return -1;
    CUptiResult r = p_subscribe(&sub, (CUpti_CallbackFunc)cb, 0);
    if (r != CUPTI_SUCCESS) return (int)r;
    p_enable(1, sub, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED);
    static const CUpti_CallbackId load_ids[] = {
        CUPTI_DRIVER_TRACE_CBID_cuModuleLoad,          CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData,
        CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,    CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary,
        CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData,     CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile,
    };
    for (size_t i = 0; i < sizeof load_ids / sizeof *load_ids; i++)
        p_enable(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, load_ids[i]);
    inited = 1;
    return 0;
}

int ptxvcap_start(const char *dir, int snapshots, unsigned long long max_alloc_bytes,
                  unsigned long long budget_bytes) {
    pthread_mutex_lock(&mu);
    if (!inited || active) {
        pthread_mutex_unlock(&mu);
        return -1;
    }
    snprintf(cap_dir, sizeof cap_dir, "%s", dir);
    char p[4200];
    snprintf(p, sizeof p, "%s/snap", dir);
    mkdir(dir, 0755);
    mkdir(p, 0755);
    snap_mode = snapshots;
    max_alloc = max_alloc_bytes;
    snap_budget = budget_bytes;
    seq = 0;
    graph_launches = skipped_snaps = 0;
    nkeys = 0;
    tag[0] = 0;
    active = 1;
    for (size_t i = 0; i < sizeof launch_ids / sizeof *launch_ids; i++)
        p_enable(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, launch_ids[i]);
    pthread_mutex_unlock(&mu);
    return 0;
}

// Stops recording; returns the number of launches recorded.
int ptxvcap_stop(unsigned long long *graph, unsigned long long *skipped) {
    pthread_mutex_lock(&mu);
    if (active)
        for (size_t i = 0; i < sizeof launch_ids / sizeof *launch_ids; i++)
            p_enable(0, sub, CUPTI_CB_DOMAIN_DRIVER_API, launch_ids[i]);
    active = 0;
    if (graph) *graph = graph_launches;
    if (skipped) *skipped = skipped_snaps;
    int n = (int)seq;
    pthread_mutex_unlock(&mu);
    return n;
}

// Suspend (0) or restore snapshots while a CUDA graph is being captured on any stream: the
// context synchronization a snapshot needs is illegal then and invalidates the capture.
// Returns the previous mode.
int ptxvcap_set_snap(int mode) {
    pthread_mutex_lock(&mu);
    int old = snap_mode;
    snap_mode = mode;
    pthread_mutex_unlock(&mu);
    return old;
}

void ptxvcap_set_tag(const char *t) {
    pthread_mutex_lock(&mu);
    snprintf(tag, sizeof tag, "%s", t ? t : "");
    pthread_mutex_unlock(&mu);
}

// Hashes (16 hex digits, one per line) of the modules this process loaded; returns the number
// of bytes needed.
size_t ptxvcap_modules(char *buf, size_t cap) {
    pthread_mutex_lock(&mu);
    size_t need = 17 * nmods + 1;
    if (buf && cap >= need) {
        char *p = buf;
        for (size_t i = 0; i < nmods; i++) p += sprintf(p, "%016llx\n", mods[i]);
        *p = 0;
    }
    pthread_mutex_unlock(&mu);
    return need;
}
