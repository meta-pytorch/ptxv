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
// - Launch order (docs/capture.md "Launch order"): each launch also records its stream's id
//   (cuStreamGetId), whether it went into a graph capture (and the capture's id), its
//   ProgrammaticEvent / LaunchCompletionEvent events, its memory synchronization domain and
//   "oseq", its place among the launches and stream order calls; event records, event waits,
//   rewritten capture dependencies, event destructions, IPC event opens and exports, and graph
//   launches go to <dir>/events.jsonl with an "oseq" from the same counter. Every one is
//   written at API_EXIT, only when the call succeeded, with the calling thread ("tid") and the
//   times at API_ENTER and API_EXIT ("t0", "t1"): the records of several threads are in the
//   order the driver took the calls only when the calls did not overlap in time.
//   python/ptxv/_order.py turns them into the case's predecessor relations.
// - cuGraphLaunch is counted (graph_launches) and logged as a stream order call: a graph's
//   kernels are captured only when the graph itself was captured inside the region.
#define _GNU_SOURCE
#include <cuda.h>
#include <cupti.h>  // includes generated_cuda_meta.h (the launch parameter structs)
#include <dlfcn.h>
#include <execinfo.h>
#include <limits.h>
#include <pthread.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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
typedef CUresult (*sid_t)(CUstream, unsigned long long *);
typedef CUresult (*scap_t)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
typedef CUresult (*sattr_t)(CUstream, CUlaunchAttributeID, CUlaunchAttributeValue *);
typedef CUresult (*sdev_t)(CUstream, CUdevice *);
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
// Legacy ([0]) and per-thread default stream ([1]) flavours: a stream query takes the
// flavour of the call it describes (stream 0 is a different stream in each).
static sid_t c_sid[2];
static scap_t c_scap[2];
static sattr_t c_sattr[2];
static sdev_t c_sdev[2];

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
static unsigned long long oseq;        // launches and stream order calls recorded since then
static int order_calls = 1;            // record stream order calls (injection: PTXV_CAPTURE_ORDER)
static unsigned long long inj_unwritten;  // launches seen but not written since the last written one
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
// Per-thread: the record of the launch in flight (built at API_ENTER, written at API_EXIT if
// the launch succeeded), the launches left unwritten that it accounts for (injection mode),
// and the time at the end of API_ENTER of the launch or stream order call in flight.
static __thread char *t_rec;
static __thread size_t t_rec_len;
static __thread unsigned long long t_unrec;
static __thread unsigned long long t_t0;

// Host time (CLOCK_MONOTONIC, ns) and the calling thread (gettid).
static unsigned long long now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}
static long tid(void) { return (long)syscall(SYS_gettid); }

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

static void json_str(FILE *f, const char *s);

// <hash>.json next to a module image: the shared objects on the stack of the load (the first
// one outside the CUDA driver, runtime, CUPTI and this library is the loader: libcublasLt,
// libcudnn_engines_*, libnccl, libtorch_cuda, a Python extension, the main program). This is
// the module's provenance (library and version from its path) for vendor kernels.
static void write_loader(const char *path) {
    void *fr[64];
    int n = backtrace(fr, 64);
    char tmp[4300];
    snprintf(tmp, sizeof tmp, "%s.%d.tmp", path, (int)getpid());
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    const char *seen[64];
    int ns = 0, loader = 0;
    fputs("{\"objects\": [", f);
    for (int i = 0; i < n; i++) {
        Dl_info di;
        if (!dladdr(fr[i], &di) || !di.dli_fname) continue;
        int dup = 0;
        for (int j = 0; j < ns; j++)
            if (!strcmp(seen[j], di.dli_fname)) dup = 1;
        if (dup || ns == 64) continue;
        seen[ns++] = di.dli_fname;
        fputs(ns > 1 ? ", " : "", f);
        json_str(f, di.dli_fname);
    }
    fputs("], \"loader\": ", f);
    for (int j = 0; j < ns && !loader; j++) {
        const char *b = strrchr(seen[j], '/');
        b = b ? b + 1 : seen[j];
        if (strstr(b, "libcuda.so") || strstr(b, "libcupti") || strstr(b, "ptxvcap") ||
            strstr(b, "libcudart") || strstr(b, "libc.so") || strstr(b, "libpthread"))
            continue;
        char rp[PATH_MAX];
        json_str(f, seen[j]);
        fputs(", \"loader_real\": ", f);
        json_str(f, realpath(seen[j], rp) ? rp : seen[j]);
        loader = 1;
    }
    if (!loader) fputs("null", f);
    fputs("}\n", f);
    fclose(f);
    rename(tmp, path);
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
        snprintf(path, sizeof path, "%s/%016llx.json", mod_dir, h);
        if (stat(path, &st) != 0) write_loader(path);
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

// ---------------------------------------------------------------- injection mode
// CUDA_INJECTION64_PATH=<this library> (InitializeInjection at the end): records launches of
// any process from cuInit on, selected by kernel-name regex and invocation index, with no
// Python in the process (docs/capture.md "Any process"). State below is guarded by mu.
static int inj;                    // injection mode active
static regex_t inj_re, inj_skip;   // PTXV_CAPTURE_FILTER / PTXV_CAPTURE_SKIP
static int inj_has_re, inj_has_skip;
#define INJ_MAXR 64
static unsigned inj_lo[INJ_MAXR], inj_hi[INJ_MAXR];  // PTXV_CAPTURE_INDEX ranges (inclusive)
static int inj_nr;
static unsigned inj_max = 16;      // PTXV_CAPTURE_MAX: snapshotted launches in total
static unsigned inj_snapped;
static int inj_record_all;         // PTXV_CAPTURE_RECORD=matching: also unselected matches
static unsigned long long inj_matched, inj_seen;
// Per kernel name: invocations of matching launches so far (the invocation index).
static unsigned long long *inv_key;
static unsigned *inv_cnt;
static size_t ninv, capinv;
// Fatbin images handed to module / library loads (PTXV_CAPTURE_FATBIN, default on): load
// handle -> image pointer. When a selected launch's library / module handle has a fatbin
// image, the fatbin is written to <mod_dir>/<fnv of its bytes>.fatbin and the launch record
// names it ("fatbin"); the fatbin may embed the kernels' PTX. Fatbins registered by the CUDA
// runtime live in the binary's .nv_fatbin section, so the pointer stays valid; with lazy
// loading the module image itself only arrives inside the first launch.
#define MAXFAT 65536
static void *fat_h[MAXFAT];
static const void *fat_ptr[MAXFAT];
static size_t nfat;
static int inj_fatbin = 1;
static __thread const void *t_call_img;

static unsigned inv_next(const char *nm) {
    unsigned long long k = fnv(nm, strlen(nm), 0xcbf29ce484222325ull);
    for (size_t i = 0; i < ninv; i++)
        if (inv_key[i] == k) return inv_cnt[i]++;
    if (ninv == capinv) {
        capinv = capinv ? 2 * capinv : 256;
        inv_key = realloc(inv_key, capinv * sizeof *inv_key);
        inv_cnt = realloc(inv_cnt, capinv * sizeof *inv_cnt);
    }
    inv_key[ninv] = k;
    inv_cnt[ninv++] = 1;
    return 0;
}

static int inj_index_selected(unsigned i) {
    if (!inj_nr) return i == 0;
    for (int r = 0; r < inj_nr; r++)
        if (inj_lo[r] <= i && i <= inj_hi[r]) return 1;
    return 0;
}

// Start and size of a fatbin image (a __fatBinC_Wrapper_t or a bare fatbin), or 0.
static size_t fatbin_extent(const void *p, const void **start) {
    if (!p) return 0;
    unsigned w0;
    memcpy(&w0, p, 4);
    if (w0 == 0x466243b1u) p = *(const void *const *)((const char *)p + 8);  // wrapper
    if (!p) return 0;
    const unsigned char *b = p;
    unsigned magic;
    unsigned short hsz;
    unsigned long long fsz;
    memcpy(&magic, b, 4);
    if (magic != 0xba55ed50u) return 0;
    memcpy(&hsz, b + 6, 2);
    memcpy(&fsz, b + 8, 8);
    if (hsz < 16 || fsz > (1ull << 32)) return 0;
    *start = p;
    return hsz + fsz;
}

static void fat_note(void *h, const void *img) {
    const void *st;
    if (!inj || !inj_fatbin || !h || !fatbin_extent(img, &st)) return;
    for (size_t i = 0; i < nfat; i++)
        if (fat_h[i] == h) {
            fat_ptr[i] = img;
            return;
        }
    if (nfat < MAXFAT) {
        fat_h[nfat] = h;
        fat_ptr[nfat++] = img;
    }
}

// Writes the fatbin of load handle h (once, content-addressed); returns its hash or 0.
static unsigned long long fat_dump(void *h) {
    for (size_t i = 0; i < nfat; i++) {
        if (fat_h[i] != h) continue;
        const void *st;
        size_t n = fatbin_extent(fat_ptr[i], &st);
        if (!n) return 0;
        unsigned long long fh = fnv(st, n, 0xcbf29ce484222325ull);
        char path[4200], tmp[4300];
        snprintf(path, sizeof path, "%s/%016llx.fatbin", mod_dir, fh);
        struct stat sb;
        if (stat(path, &sb) != 0) {
            snprintf(tmp, sizeof tmp, "%s.%d.tmp", path, (int)getpid());
            write_file(tmp, st, n);
            rename(tmp, path);
        }
        return fh;
    }
    return 0;
}

// Record one launch (API_ENTER; written by launch_exit). Called with mu held.
static void on_launch_enter(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                            unsigned by, unsigned bz, unsigned smem, CUstream s,
                            const CUlaunchConfig *cfg, void **params, void **extra,
                            const char *sym, int cbid) {
    t_snap = 0;
    free(t_rec);
    t_rec = 0;
    t_unrec = 0;
    unsigned cx = 1, cy = 1, cz = 1;
    // Preferred cluster dimension (0s without the attribute): S-CGA-SHAPE, probes/cga_size.
    unsigned px = 0, py = 0, pz = 0;
    int pdl = 0;
    // Launch order: ProgrammaticEvent / LaunchCompletionEvent events (0: none) and the memory
    // synchronization domain attributes (-1: the stream's).
    unsigned long long pev = 0, cev = 0;
    int pev_start = 0, msd = -1, msdm0 = -1, msdm1 = -1;
    int ptsz = cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz;
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
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT) {
                pev = (unsigned long long)cfg->attrs[i].value.programmaticEvent.event;
                pev_start = cfg->attrs[i].value.programmaticEvent.triggerAtBlockStart != 0;
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT) {
                cev = (unsigned long long)cfg->attrs[i].value.launchCompletionEvent.event;
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN) {
                msd = (int)cfg->attrs[i].value.memSyncDomain;
            } else if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP) {
                msdm0 = cfg->attrs[i].value.memSyncDomainMap.default_;
                msdm1 = cfg->attrs[i].value.memSyncDomainMap.remote;
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
    // Injection mode: kernel-name filter, then the invocation index among the matching
    // launches of this name.
    int inj_sel = 0;
    long inv = -1;
    if (inj) {
        inj_seen++;
        if ((inj_has_re && regexec(&inj_re, nm, 0, 0, 0)) ||
            (inj_has_skip && !regexec(&inj_skip, nm, 0, 0, 0))) {
            inj_unwritten++;
            return;
        }
        inj_matched++;
        inv = inv_next(nm);
        inj_sel = inj_index_selected((unsigned)inv) && inj_snapped < inj_max;
        if (!inj_sel && !inj_record_all) {
            inj_unwritten++;
            return;
        }
    }
    FILE *fp = open_memstream(&t_rec, &t_rec_len);
    if (!fp) return;
    // Whether the launch goes into a graph capture (it runs when the graph is launched, not
    // now), and the capture's id; an invalidated capture is still one. -1: unknown.
    CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
    cuuint64_t cap_id = 0;
    int capturing = -1;
    if (c_scap[ptsz] && c_scap[ptsz](s, &cs, &cap_id) == CUDA_SUCCESS)
        capturing = cs != CU_STREAM_CAPTURE_STATUS_NONE;
    else if (c_iscap && c_iscap(s, &cs) == CUDA_SUCCESS)
        capturing = cs != CU_STREAM_CAPTURE_STATUS_NONE ? -1 : 0;  // no id: unknown capture
    CUdevice dev = -1;
    if (c_getdev) c_getdev(&dev);
    t_seq = seq++;
    fprintf(fp, "\"name\": ");
    json_str(fp, nm);
    fprintf(fp, ", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], \"smem\": %u, "
                "\"cluster\": [%u, %u, %u], \"pdl\": %d, \"stream\": %llu, \"dev\": %d, "
                "\"cbid\": %d, \"capturing\": %d, \"capture_id\": %llu, \"ptsz\": %d, ",
            gx, gy, gz, bx, by, bz, smem, cx, cy, cz, pdl, (unsigned long long)s, (int)dev, cbid,
            capturing, capturing == 1 ? (unsigned long long)cap_id : 0ull, ptsz);
    {
        // Not on a capturing stream: cuStreamGetId there fails and invalidates the capture.
        unsigned long long sid;
        if (capturing == 0 && c_sid[ptsz] && c_sid[ptsz](s, &sid) == CUDA_SUCCESS)
            fprintf(fp, "\"stream_id\": %llu, ", sid);
        // Memory synchronization domain: the logical one (launch, else stream) through the map
        // (launch, else stream; default -> 0, remote -> 1 by default). S-SCOPE.
        CUlaunchAttributeValue v;
        if (msd < 0) {
            msd = 0;
            memset(&v, 0, sizeof v);
            if (c_sattr[ptsz] && c_sattr[ptsz](s, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, &v) == CUDA_SUCCESS)
                msd = (int)v.memSyncDomain;
        }
        if (msdm0 < 0) {
            msdm0 = 0;
            msdm1 = 1;
            memset(&v, 0, sizeof v);
            if (c_sattr[ptsz] &&
                c_sattr[ptsz](s, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &v) == CUDA_SUCCESS) {
                msdm0 = v.memSyncDomainMap.default_;
                msdm1 = v.memSyncDomainMap.remote;
            }
        }
        fprintf(fp, "\"mem_sync_domain\": %d, ", msd == CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE ? msdm1 : msdm0);
        if (pev)
            fprintf(fp, "\"prog_event\": {\"event\": %llu, \"trigger_at_block_start\": %d}, ", pev,
                    pev_start);
        if (cev) fprintf(fp, "\"completion_event\": %llu, ", cev);
        // Injection mode: launches seen but not written since the previous written one (their
        // order is then not in the records).
        if (inj) {
            fprintf(fp, "\"unrecorded_before\": %llu, ", inj_unwritten);
            t_unrec = inj_unwritten;
            inj_unwritten = 0;
        }
    }
    fprintf(fp, "\"tag\": ");
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
        if (inj_sel && nfat) {
            unsigned long long fh = 0;
            if (k && lib) fh = fat_dump(lib);
            if (!fh && mod) fh = fat_dump(mod);
            if (!fh && f && !k && c_fmod && c_fmod(&mod, f) == CUDA_SUCCESS) fh = fat_dump(mod);
            if (fh) fprintf(fp, ", \"fatbin\": \"%016llx\"", fh);
        }
    }
    if (inj) fprintf(fp, ", \"inv\": %ld", inv);
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
    t_snap = snap_mode && capturing == 0;
    if (inj) {
        t_snap = t_snap && inj_sel;
        if (t_snap) inj_snapped++;
    } else if (t_snap && snap_mode == 2) {
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
            // A ragged `ptr_shift` tensor map (CuTe create_ragged_tensor_for_tma) has its base
            // 2^30 rows before its tensor: snapshot the allocation 2^30 * stride[d] on, for
            // each dim d of extent 2^30 (the case writer gives it a negative byte_offset).
            if (!z && v && o % 64 == 0 && o + 128 <= plen) {
                unsigned w2, hi;
                memcpy(&w2, pbuf + o + 8, 4);
                memcpy(&hi, pbuf + o + 0x1C, 4);
                unsigned rank = ((w2 >> 4) & 7) + 1;
                if (!(w2 & 0xD) && !((w2 >> 11) & 3) && !(w2 >> 22) && rank <= 5)
                    for (unsigned d = 1; d < rank; d++) {
                        unsigned dim, lo;
                        memcpy(&dim, pbuf + o + 0x20 + 4 * d, 4);
                        if (dim + 1ull != (1ull << 30)) continue;
                        memcpy(&lo, pbuf + o + 8 + 4 * d, 4);
                        unsigned long long st = ((unsigned long long)lo << 4) |
                                                ((unsigned long long)((hi >> (4 * (d - 1))) & 15) << 36);
                        total += add_target(v + (st << 30));
                    }
            }
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
    fclose(fp);
    if (t_snap) {
        c_sync();
        snap("pre");
    }
}

// API_EXIT of a launch: its record goes to <dir>/launches.jsonl with the next "oseq" if the
// launch succeeded; a failed launch leaves no record (the launches it accounted for as
// unwritten carry over to the next record).
static void launch_exit(int ok, unsigned long long t1) {
    if (!t_rec) return;
    pthread_mutex_lock(&mu);
    if (active && ok) {
        char path[4200];
        snprintf(path, sizeof path, "%s/launches.jsonl", cap_dir);
        FILE *fp = fopen(path, "a");
        if (fp) {
            fprintf(fp, "{\"seq\": %u, \"oseq\": %llu, \"tid\": %ld, \"t0\": %llu, \"t1\": %llu, %s}\n",
                    t_seq, oseq++, tid(), t_t0, t1, t_rec);
            fclose(fp);
        }
    } else if (!ok) {
        inj_unwritten += t_unrec;
    }
    pthread_mutex_unlock(&mu);
    free(t_rec);
    t_rec = 0;
}

// A stream order call that succeeded (API_EXIT): event record, event wait, graph launch or a
// capturing stream's rewritten dependencies (cuStreamUpdateCaptureDependencies), one line in
// <dir>/events.jsonl with the stream fields of a launch record and the next "oseq";
// event destruction (the handle no longer names the event recorded under it) or IPC event
// open or export (another process can record the event), with no stream.
static void on_order_call(CUpti_CallbackId id, const void *params, unsigned long long t0,
                          unsigned long long t1) {
    const char *kind = "record", *what = "event";
    unsigned long long h = 0;
    CUstream s = 0;
    unsigned fl = 0;
    int ptsz = 0, nostream = 0;
    switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord: {
        const cuEventRecord_params *p = params;
        h = (unsigned long long)p->hEvent;
        s = p->hStream;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags: {
        const cuEventRecordWithFlags_params *p = params;
        h = (unsigned long long)p->hEvent;
        s = p->hStream;
        fl = p->flags;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent: {
        const cuStreamWaitEvent_params *p = params;
        kind = "wait";
        h = (unsigned long long)p->hEvent;
        s = p->hStream;
        fl = p->Flags;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch: {
        const cuGraphLaunch_params *p = params;
        kind = "graph_launch";
        what = "graph";
        h = (unsigned long long)p->hGraph;
        s = p->hStream;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies: {
        const cuStreamUpdateCaptureDependencies_params *p = params;
        kind = "capture_deps";
        what = "nodes";
        h = p->numDependencies;
        s = p->hStream;
        fl = p->flags;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2_ptsz:
        ptsz = 1;
        /* fallthrough */
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2: {
        const cuStreamUpdateCaptureDependencies_v2_params *p = params;
        kind = "capture_deps";
        what = "nodes";
        h = p->numDependencies;
        s = p->hStream;
        fl = p->flags;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy: {
        const cuEventDestroy_params *p = params;
        kind = "destroy";
        h = (unsigned long long)p->hEvent;
        nostream = 1;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2: {
        const cuEventDestroy_v2_params *p = params;
        kind = "destroy";
        h = (unsigned long long)p->hEvent;
        nostream = 1;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle: {
        const cuIpcOpenEventHandle_params *p = params;
        kind = "ipc_open";
        h = p->phEvent ? (unsigned long long)*p->phEvent : 0;
        nostream = 1;
        break;
    }
    case CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle: {
        const cuIpcGetEventHandle_params *p = params;
        kind = "ipc_get";
        h = (unsigned long long)p->event;
        nostream = 1;
        break;
    }
    default:
        return;
    }
    pthread_mutex_lock(&mu);
    if (!active || !order_calls) {
        pthread_mutex_unlock(&mu);
        return;
    }
    char path[4200];
    snprintf(path, sizeof path, "%s/events.jsonl", cap_dir);
    FILE *fp = fopen(path, "a");
    if (fp && nostream) {
        CUdevice dev = -1;
        if (c_getdev) c_getdev(&dev);
        fprintf(fp, "{\"oseq\": %llu, \"ev\": \"%s\", \"event\": %llu, \"dev\": %d, \"tid\": %ld, "
                    "\"t0\": %llu, \"t1\": %llu}\n",
                oseq++, kind, h, (int)dev, tid(), t0, t1);
        fclose(fp);
    } else if (fp) {
        CUstreamCaptureStatus cs = CU_STREAM_CAPTURE_STATUS_NONE;
        cuuint64_t cap_id = 0;
        int capturing = -1;
        if (c_scap[ptsz] && c_scap[ptsz](s, &cs, &cap_id) == CUDA_SUCCESS)
            capturing = cs != CU_STREAM_CAPTURE_STATUS_NONE;
        // cuStreamGetId / cuStreamGetDevice only on a stream not capturing (launchcap.c).
        CUdevice dev = -1;
        if (!(capturing == 0 && c_sdev[ptsz] && c_sdev[ptsz](s, &dev) == CUDA_SUCCESS) && c_getdev)
            c_getdev(&dev);
        fprintf(fp, "{\"oseq\": %llu, \"ev\": \"%s\", \"%s\": %llu, \"flags\": %u, \"stream\": %llu, "
                    "\"ptsz\": %d, \"dev\": %d, \"capturing\": %d, \"capture_id\": %llu",
                oseq++, kind, what, h, fl, (unsigned long long)s, ptsz, (int)dev, capturing,
                capturing == 1 ? (unsigned long long)cap_id : 0ull);
        unsigned long long sid;
        if (capturing == 0 && c_sid[ptsz] && c_sid[ptsz](s, &sid) == CUDA_SUCCESS)
            fprintf(fp, ", \"stream_id\": %llu", sid);
        fprintf(fp, ", \"tid\": %ld, \"t0\": %llu, \"t1\": %llu}\n", tid(), t0, t1);
        fclose(fp);
    }
    if (id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz)
        graph_launches++;
    pthread_mutex_unlock(&mu);
}

// Stream order calls and graph launches (cb: logged at API_EXIT if they succeeded).
static int order_cbid(CUpti_CallbackId id) {
    switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags:
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy:
    case CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle:
    case CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle:
        return 1;
    default:
        return 0;
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
            t_call_img = 0;
            // The image argument (second member of each of these parameter structs, except
            // the file-name loads).
            if (inj && c->functionParams && id != CUPTI_DRIVER_TRACE_CBID_cuModuleLoad &&
                id != CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile)
                t_call_img = ((const void *const *)c->functionParams)[1];
        } else if (c->functionParams && t_call_img) {
            // Every one of these takes the output handle pointer first.
            void **hp0 = *(void ***)c->functionParams;
            if (hp0 && *hp0) {
                pthread_mutex_lock(&mu);
                fat_note(*hp0, t_call_img);
                pthread_mutex_unlock(&mu);
            }
            t_call_img = 0;
        }
        if (c->callbackSite == CUPTI_API_EXIT && t_call_mod && c->functionParams) {
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
    // Stream order calls and graph launches: logged at API_EXIT if they succeeded.
    if (order_cbid(id)) {
        if (c->callbackSite == CUPTI_API_ENTER) {
            t_t0 = now_ns();
            return;
        }
        unsigned long long t1 = now_ns();
        if (!c->functionReturnValue || *(const CUresult *)c->functionReturnValue != CUDA_SUCCESS) return;
        if (id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch ||
            id == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz) {
            pthread_mutex_lock(&mu);
            if (active && !order_calls) graph_launches++;
            pthread_mutex_unlock(&mu);
        }
        // (counts a graph launch when order calls are recorded)
        on_order_call(id, c->functionParams, t_t0, t1);
        return;
    }
    if (c->callbackSite == CUPTI_API_ENTER) {
        t_call_mod = 0;
        pthread_mutex_lock(&mu);
        if (!active) {
            pthread_mutex_unlock(&mu);
            t_snap = 0;
            free(t_rec);
            t_rec = 0;
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
        t_t0 = now_ns();
    } else {
        unsigned long long t1 = now_ns();
        int ok = c->functionReturnValue && *(const CUresult *)c->functionReturnValue == CUDA_SUCCESS;
        launch_exit(ok, t1);
        if (t_snap) {
            c_sync();
            snap("post");
            t_snap = 0;
            pthread_mutex_unlock(&mu);
        }
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
// Stream order calls (with order_calls).
static const CUpti_CallbackId order_ids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuEventRecord,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags,
    CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent,
    CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuEventDestroy,
    CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2,
    CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle,
    CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2,
    CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2_ptsz,
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
    c_sid[0] = (sid_t)dlsym(cu, "cuStreamGetId");
    c_sid[1] = (sid_t)dlsym(cu, "cuStreamGetId_ptsz");
    // The first version (status and id), whatever cuda.h maps the name to now.
    c_scap[0] = (scap_t)dlsym(cu, "cuStreamGetCaptureInfo");
    c_scap[1] = (scap_t)dlsym(cu, "cuStreamGetCaptureInfo_ptsz");
    c_sattr[0] = (sattr_t)dlsym(cu, "cuStreamGetAttribute");
    c_sattr[1] = (sattr_t)dlsym(cu, "cuStreamGetAttribute_ptsz");
    c_sdev[0] = (sdev_t)dlsym(cu, "cuStreamGetDevice");
    c_sdev[1] = (sdev_t)dlsym(cu, "cuStreamGetDevice_ptsz");
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
    oseq = 0;
    inj_unwritten = 0;
    graph_launches = skipped_snaps = 0;
    nkeys = 0;
    tag[0] = 0;
    active = 1;
    for (size_t i = 0; i < sizeof launch_ids / sizeof *launch_ids; i++)
        p_enable(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, launch_ids[i]);
    if (order_calls)
        for (size_t i = 0; i < sizeof order_ids / sizeof *order_ids; i++)
            p_enable(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, order_ids[i]);
    pthread_mutex_unlock(&mu);
    return 0;
}

// Stops recording; returns the number of launches recorded.
int ptxvcap_stop(unsigned long long *graph, unsigned long long *skipped) {
    pthread_mutex_lock(&mu);
    if (active) {
        for (size_t i = 0; i < sizeof launch_ids / sizeof *launch_ids; i++)
            p_enable(0, sub, CUPTI_CB_DOMAIN_DRIVER_API, launch_ids[i]);
        for (size_t i = 0; i < sizeof order_ids / sizeof *order_ids; i++)
            p_enable(0, sub, CUPTI_CB_DOMAIN_DRIVER_API, order_ids[i]);
    }
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

// ---------------------------------------------------------------- CUDA_INJECTION64_PATH
// Environment (read once, at cuInit):
//   PTXV_CAPTURE_DIR     output directory (required); "%r" -> $RANK, else p<pid>. Holds
//                        launches.jsonl, snap/, modules/ (<hash>.cubin, <hash>.fatbin),
//                        process.json, summary.json (written at exit).
//   PTXV_CAPTURE_FILTER  POSIX extended regex on the kernel name (mangled); default: all.
//   PTXV_CAPTURE_SKIP    regex of names to leave out.
//   PTXV_CAPTURE_INDEX   invocation indices to snapshot, per kernel name, among matching
//                        launches: "0" (default), "3", "0,5-7", "all".
//   PTXV_CAPTURE_MAX     snapshotted launches in total (default 16).
//   PTXV_CAPTURE_RECORD  "selected" (default: only snapshotted launches are written) or
//                        "matching" (every matching launch, unselected ones without snapshot).
//   PTXV_CAPTURE_MAX_ALLOC, PTXV_CAPTURE_BUDGET   bytes (k/M/G suffix): largest allocation
//                        copied (default 2G) and total snapshot bytes (default 32G).
//   PTXV_CAPTURE_FATBIN  0: do not write fatbins (embedded PTX) of the used modules.
//   PTXV_CAPTURE_ORDER   1: also record stream order calls (events.jsonl; launch order of a
//                        multi-launch case, which then needs every launch written:
//                        PTXV_CAPTURE_RECORD=matching and no filter).
//   PTXV_CUPTI           libcupti to load (default libcupti.so.13, then the CUDA 13 toolkit).
static char inj_dir[4096];

static unsigned long long env_size(const char *name, unsigned long long dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    char *end;
    double x = strtod(v, &end);
    switch (*end) {
    case 'k': case 'K': x *= 1024.0; break;
    case 'm': case 'M': x *= 1024.0 * 1024; break;
    case 'g': case 'G': x *= 1024.0 * 1024 * 1024; break;
    case 't': case 'T': x *= 1024.0 * 1024 * 1024 * 1024; break;
    }
    return (unsigned long long)x;
}

/* toolchain.json: the CUDA-related shared objects mapped at exit (CUDA runtime, cutlass-dsl
 * flavor, NVRTC / nvJitLink, torch, Triton, vendor libraries), from which the converter derives
 * the toolchain provenance (cutlass-dsl backend cu12 / cu13, cudart major, ...). */
static void inj_toolchain(void) {
    static const char *pats[] = {"libcudart.so", "libcute_dsl_runtime", "_cutlass_ir.cu", "libnvrtc",
                                 "libnvJitLink", "libtorch_cuda", "libtriton", "libcublas", "libcudnn",
                                 "libnccl", "libcuda.so"};
    char path[4200];
    snprintf(path, sizeof path, "%s/toolchain.json", inj_dir);
    FILE *m = fopen("/proc/self/maps", "r");
    if (!m) return;
    FILE *f = fopen(path, "w");
    if (!f) { fclose(m); return; }
    fputs("{\"libs\": [", f);
    char line[8192], seen[64][1024];
    int ns = 0;
    while (fgets(line, sizeof line, m)) {
        char *p = strchr(line, '/');
        if (!p) continue;
        p[strcspn(p, "\n")] = 0;
        int hit = 0;
        for (size_t i = 0; i < sizeof pats / sizeof *pats && !hit; i++) hit = strstr(p, pats[i]) != NULL;
        if (!hit) continue;
        int dup = 0;
        for (int i = 0; i < ns && !dup; i++) dup = strcmp(seen[i], p) == 0;
        if (dup || ns >= 64) continue;
        snprintf(seen[ns], sizeof seen[ns], "%s", p);
        if (ns) fputs(", ", f);
        json_str(f, p);
        ns++;
    }
    fputs("]}\n", f);
    fclose(f);
    fclose(m);
}

static void inj_summary(void) {
    inj_toolchain();
    char path[4200];
    unsigned long long g = 0, sk = 0;
    int n = ptxvcap_stop(&g, &sk);
    snprintf(path, sizeof path, "%s/summary.json", inj_dir);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\"recorded\": %d, \"snapshotted\": %u, \"matched\": %llu, \"seen\": %llu, "
               "\"graph_launches\": %llu, \"skipped_snapshots\": %llu, \"modules\": %zu}\n",
            n, inj_snapped, inj_matched, inj_seen, g, sk, nmods);
    fclose(f);
}

static void inj_process_json(void) {
    char path[4200];
    snprintf(path, sizeof path, "%s/process.json", inj_dir);
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\"pid\": %d, \"argv\": [", (int)getpid());
    FILE *c = fopen("/proc/self/cmdline", "rb");
    if (c) {
        char buf[8192];
        size_t n = fread(buf, 1, sizeof buf - 1, c);
        fclose(c);
        buf[n] = 0;
        size_t i = 0;
        int first = 1;
        while (i < n) {
            if (!first) fputs(", ", f);
            json_str(f, buf + i);
            first = 0;
            i += strlen(buf + i) + 1;
        }
    }
    fputs("], \"env\": {", f);
    static const char *keys[] = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES",
                                 "PTXV_CAPTURE_FILTER", "PTXV_CAPTURE_SKIP", "PTXV_CAPTURE_INDEX",
                                 "PTXV_CAPTURE_MAX", "PTXV_CAPTURE_RECORD", "CUDA_MODULE_LOADING"};
    int first = 1;
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        const char *v = getenv(keys[i]);
        if (!v) continue;
        fprintf(f, "%s\"%s\": ", first ? "" : ", ", keys[i]);
        json_str(f, v);
        first = 0;
    }
    char cwd[4096];
    fputs("}, \"cwd\": ", f);
    json_str(f, getcwd(cwd, sizeof cwd) ? cwd : "");
    fputs("}\n", f);
    fclose(f);
}

int InitializeInjection(void) {
    const char *d = getenv("PTXV_CAPTURE_DIR");
    if (!d || !*d) {
        fprintf(stderr, "ptxvcap: CUDA_INJECTION64_PATH set but PTXV_CAPTURE_DIR is not; not capturing\n");
        return 1;
    }
    const char *at = strstr(d, "%r");
    if (at) {
        char rk[64];
        const char *r = getenv("RANK");
        if (r) snprintf(rk, sizeof rk, "%s", r);
        else snprintf(rk, sizeof rk, "p%d", (int)getpid());
        snprintf(inj_dir, sizeof inj_dir, "%.*s%s%s", (int)(at - d), d, rk, at + 2);
    } else {
        snprintf(inj_dir, sizeof inj_dir, "%s", d);
    }
    mkdir(inj_dir, 0755);
    char mdir[4200];
    snprintf(mdir, sizeof mdir, "%s/modules", inj_dir);
    mkdir(mdir, 0755);
    const char *re = getenv("PTXV_CAPTURE_FILTER");
    if (re && *re) {
        if (regcomp(&inj_re, re, REG_EXTENDED | REG_NOSUB)) {
            fprintf(stderr, "ptxvcap: bad PTXV_CAPTURE_FILTER regex %s\n", re);
            return 1;
        }
        inj_has_re = 1;
    }
    const char *sk = getenv("PTXV_CAPTURE_SKIP");
    if (sk && *sk) {
        if (regcomp(&inj_skip, sk, REG_EXTENDED | REG_NOSUB)) {
            fprintf(stderr, "ptxvcap: bad PTXV_CAPTURE_SKIP regex %s\n", sk);
            return 1;
        }
        inj_has_skip = 1;
    }
    const char *ix = getenv("PTXV_CAPTURE_INDEX");
    if (ix && !strcmp(ix, "all")) {
        inj_lo[0] = 0;
        inj_hi[0] = ~0u;
        inj_nr = 1;
    } else if (ix && *ix) {
        const char *p = ix;
        while (*p && inj_nr < INJ_MAXR) {
            char *e;
            unsigned lo = (unsigned)strtoul(p, &e, 10), hi = lo;
            if (e == p) break;
            if (*e == '-') {
                p = e + 1;
                hi = (unsigned)strtoul(p, &e, 10);
            }
            inj_lo[inj_nr] = lo;
            inj_hi[inj_nr++] = hi;
            p = *e == ',' ? e + 1 : e;
            if (*e && *e != ',') break;
        }
    }
    inj_max = (unsigned)env_size("PTXV_CAPTURE_MAX", 16);
    const char *rec = getenv("PTXV_CAPTURE_RECORD");
    inj_record_all = rec && !strcmp(rec, "matching");
    const char *fb = getenv("PTXV_CAPTURE_FATBIN");
    inj_fatbin = !(fb && !strcmp(fb, "0"));
    // Stream order calls for the whole process only when asked (one case per launch needs none).
    const char *oc = getenv("PTXV_CAPTURE_ORDER");
    order_calls = oc && !strcmp(oc, "1");
    const char *cp = getenv("PTXV_CUPTI");
    int rc = -1;
    const char *cands[] = {cp, "libcupti.so.13", "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcupti.so.13",
                           "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcupti.so.13",
                           "/usr/local/cuda-13.0/extras/CUPTI/lib64/libcupti.so.13",
                           "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"};
    for (size_t i = 0; i < sizeof cands / sizeof *cands && rc == -1; i++)
        if (cands[i]) rc = ptxvcap_init(cands[i], mdir);
    if (rc != 0) {
        fprintf(stderr, "ptxvcap: CUPTI init failed (%d); not capturing\n", rc);
        return 1;
    }
    inj = 1;
    inj_process_json();
    if (ptxvcap_start(inj_dir, 1, env_size("PTXV_CAPTURE_MAX_ALLOC", 2ull << 30),
                      env_size("PTXV_CAPTURE_BUDGET", 32ull << 30)) != 0) {
        fprintf(stderr, "ptxvcap: start failed; not capturing\n");
        return 1;
    }
    atexit(inj_summary);
    if (getenv("PTXV_CAPTURE_VERBOSE"))
        fprintf(stderr, "ptxvcap: capturing to %s\n", inj_dir);
    return 1;
}
