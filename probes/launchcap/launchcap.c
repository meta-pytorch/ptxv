/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LD_PRELOAD launch capture: for every cuLaunchKernel / cuLaunchKernelEx, append the grid,
// block, dynamic smem, cluster dims and every parameter's bytes (sizes from
// cuFuncGetParamInfo) to $LAUNCHCAP_OUT as one JSON object per line. Used to turn a JIT
// framework's launch (CuTeDSL, binary libraries) into a ptxv case. Each launch also carries
// what its order depends on (stream id, graph capture, ProgrammaticEvent /
// LaunchCompletionEvent, memory synchronization domain); LAUNCHCAP_ORDER=1 adds the event
// records, event waits, event destructions and graph launches between them (docs/capture.md
// "Launch order"). A call is written once it returned, and only when it succeeded, with the
// calling thread and the times just before the call and after it returned.
//   gcc -shared -fPIC -O2 -o launchcap.so launchcap.c -ldl -lpthread -I/usr/local/cuda-13.0/include
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
// The first versions' names (cuda.h maps them to the _v2 entry points): both are hooked.
#undef cuEventDestroy
#undef cuStreamUpdateCaptureDependencies
CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies,
                                                   size_t numDependencies, unsigned int flags);
CUresult CUDAAPI cuStreamUpdateCaptureDependencies_ptsz(CUstream hStream, CUgraphNode *dependencies,
                                                        size_t numDependencies, unsigned int flags);

typedef CUresult (*getparam_t)(CUfunction, size_t, size_t *, size_t *);
typedef CUresult (*kgetparam_t)(CUkernel, size_t, size_t *, size_t *);
typedef CUresult (*getname_t)(const char **, CUfunction);

// The state of the launch being recorded is per thread: launches from several threads must
// not take each other's attributes.
// The `extra` argument of the launch being recorded (libraries such as cuBLAS pass their
// arguments as one packed buffer: CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE).
static __thread void **cur_extra;
// Whether the launch being recorded carries the programmatic stream serialization attribute
// (PDL: it may start before the previous kernel on the stream completes).
static __thread int cur_pdl;
// The preferred cluster dimension of the launch being recorded (0s without the attribute).
// With it, the hardware launches each group of CTAs either as one cluster of this shape or as
// clusters of the regular (fallback) shape (probes/cga_size).
static __thread unsigned cur_pref[3];
// The stream of the launch being recorded (multi-process captures order launches per stream).
static __thread unsigned long long cur_stream;
// Inside a runtime launch (__cudaLaunchKernel / cudaLaunchKernelExC), which records the launch
// itself: the driver launch it makes underneath is not recorded again.
static __thread int in_rt;
// Launch order (docs/capture.md "Launch order", ASSUMPTIONS.md E-LAUNCH-ORDER). Whether the
// launch being recorded came through a per-thread default stream entry point (`_ptsz`: stream
// 0 is the calling thread's own stream, not the legacy one), the events its ProgrammaticEvent
// and LaunchCompletionEvent attributes record (0: none), and its memory synchronization
// domain attributes (-1: the stream's).
static __thread int cur_ptsz;
static __thread unsigned long long cur_pev, cur_cev;
static __thread int cur_pev_start;  // triggerAtBlockStart
static __thread int cur_msd = -1, cur_msdmap0 = -1, cur_msdmap1 = -1;

// Output path: "%r" in $LAUNCHCAP_OUT / $LAUNCHCAP_IMAGES is replaced by $RANK (torchrun), or
// by the pid when RANK is unset, so the ranks of a multi-process run write separate files.
static const char *rank_path(const char *env, const char *dflt, char *buf, size_t n) {
    const char *v = getenv(env);
    if (!v) return dflt;
    const char *at = strstr(v, "%r");
    if (!at) return v;
    char r[32];
    const char *rk = getenv("RANK");
    if (!rk) {
        snprintf(r, sizeof r, "p%d", (int)getpid());
        rk = r;
    }
    snprintf(buf, n, "%.*s%s%s", (int)(at - v), v, rk, at + 2);
    return buf;
}

// Snapshots (LAUNCHCAP_SNAP=regex, LAUNCHCAP_SNAP_DIR=dir): for a launch whose kernel name
// matches, every device allocation an 8-byte parameter word points into (cuMemGetAddressRange)
// is written before and after the kernel (the device is synchronized around it) to
// dir/s<N>_{pre,post}_<base hex>.bin, and the launch's record carries "snap": N. A capture of
// one kernel out of a sequence then has its exact inputs and outputs, whatever earlier kernels
// wrote. Only for launches recorded before they are issued: driver launches,
// _cudaLaunchKernelEx, and, with LAUNCHCAP_SNAP set, the runtime's (`<<<>>>`,
// cudaLaunchKernelExC: PyTorch's own kernels), whose function is then looked up before the
// launch (MISS 173: recorded after it, they never had a snapshot). Per thread, but for the
// snapshot numbers.
#define SNAP_MAX (1 << 16)
static __thread int snap_on;
static __thread int snap_allowed;  // set by the hooks that record before launching
static unsigned snap_seq;          // the next snapshot's number
static __thread unsigned snap_cur; // the number of this thread's snapshot in flight
static __thread unsigned char *snap_buf;  // SNAP_MAX bytes, allocated on first use
static __thread size_t snap_len;
static regex_t snap_re;
static int snap_re_ok;
static pthread_once_t snap_once = PTHREAD_ONCE_INIT;
static void snap_init(void) {
    const char *p = getenv("LAUNCHCAP_SNAP");
    snap_re_ok = p && !regcomp(&snap_re, p, REG_EXTENDED | REG_NOSUB);
}
static int snap_match(const char *nm) {
    pthread_once(&snap_once, snap_init);
    return snap_re_ok && !regexec(&snap_re, nm, 0, 0, 0);
}
// Whether snapshots were asked for at all (LAUNCHCAP_SNAP).
static int snap_wanted(void) {
    pthread_once(&snap_once, snap_init);
    return snap_re_ok;
}
static void snap_dump(const char *tag) {
    typedef CUresult (*sync_t)(void);
    typedef CUresult (*range_t)(CUdeviceptr *, size_t *, CUdeviceptr);
    typedef CUresult (*d2h_t)(void *, CUdeviceptr, size_t);
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    sync_t sync = (sync_t)dlsym(cuda, "cuCtxSynchronize");
    range_t range = (range_t)dlsym(cuda, "cuMemGetAddressRange_v2");
    d2h_t d2h = (d2h_t)dlsym(cuda, "cuMemcpyDtoH_v2");
    if (!sync || !range || !d2h) return;
    sync();
    const char *dir = getenv("LAUNCHCAP_SNAP_DIR");
    CUdeviceptr seen[512];
    unsigned ns = 0;
    for (size_t o = 0; o + 8 <= snap_len; o += 8) {
        unsigned long long v;
        memcpy(&v, snap_buf + o, 8);
        CUdeviceptr b;
        size_t z;
        if (!v || range(&b, &z, (CUdeviceptr)v) != CUDA_SUCCESS || z > (1ull << 28)) continue;
        unsigned i = 0;
        while (i < ns && seen[i] != b) i++;
        if (i < ns || ns == 512) continue;
        seen[ns++] = b;
        void *h = malloc(z);
        if (!h) continue;
        if (d2h(h, b, z) == CUDA_SUCCESS) {
            char path[4096];
            snprintf(path, sizeof path, "%s/s%u_%s_%llx.bin", dir ? dir : ".", snap_cur, tag,
                     (unsigned long long)b);
            FILE *f = fopen(path, "wb");
            if (f) {
                fwrite(h, 1, z, f);
                fclose(f);
            }
        }
        free(h);
    }
}
static void snap_param(size_t off, const void *p, size_t n) {
    if (snap_on && off + n <= SNAP_MAX) {
        memcpy(snap_buf + off, p, n);
        if (off + n > snap_len) snap_len = off + n;
    }
}

static FILE *open_out(void) {
    char buf[4096];
    return fopen(rank_path("LAUNCHCAP_OUT", "launchcap.jsonl", buf, sizeof buf), "a");
}
// One record per line, whole: threads append under out_mu (out_begin .. out_end).
static pthread_mutex_t out_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *out_begin(void) {
    pthread_mutex_lock(&out_mu);
    FILE *fp = open_out();
    if (!fp) pthread_mutex_unlock(&out_mu);
    return fp;
}
static void out_end(FILE *fp) {
    fclose(fp);
    pthread_mutex_unlock(&out_mu);
}
// The calling thread (gettid), on every launch and stream order record: the records of
// several threads are in the order the driver took the calls only when the calls did not
// overlap in time (python/ptxv/_order.py).
static long tid(void) { return (long)syscall(SYS_gettid); }

static void *(*real_dlsym)(void *, const char *);
static void init_real_dlsym(void) {
    if (!real_dlsym) real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
}

// Host time (CLOCK_MONOTONIC, ns: comparable across the processes of one box) and the
// current device ordinal: the order in which ranks submitted their work.
static unsigned long long now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}
static int cur_device(void) {
    typedef CUresult (*cgd_t)(CUdevice *);
    static cgd_t g;
    if (!g) {
        void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        init_real_dlsym();
        if (cuda) g = (cgd_t)real_dlsym(cuda, "cuCtxGetDevice");
    }
    CUdevice d = -1;
    if (g && g(&d) != CUDA_SUCCESS) d = -1;
    return (int)d;
}

// What a record says about its stream (launch order): whether work issued to it is captured
// into a graph (it then runs when the graph is launched, not now) with the capture sequence's
// id, and for a stream not capturing, cuStreamGetId's id (unique among the process's streams,
// unlike a handle, which a new stream may reuse; stream 0 resolved as the entry point does:
// the calling thread's stream through a _ptsz one, else the legacy stream) and the stream's
// device. On a capturing stream cuStreamGetId / cuStreamGetDevice fail with
// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidate the capture (GB300, CUDA 13 driver),
// so they are not made; cuStreamGetCaptureInfo and cuStreamGetAttribute are legal there.
// Each query uses the entry point of the call's own flavour.
typedef CUresult (*sid_t)(CUstream, unsigned long long *);
typedef CUresult (*scap_t)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
typedef CUresult (*sdev_t)(CUstream, CUdevice *);
typedef CUresult (*sattr_t)(CUstream, CUlaunchAttributeID, CUlaunchAttributeValue *);
static sid_t q_sid[2];
static scap_t q_scap[2];
static sdev_t q_sdev[2];
static sattr_t q_sattr[2];
static int q_init;
static void stream_queries(void) {
    if (__atomic_load_n(&q_init, __ATOMIC_ACQUIRE)) return;
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    init_real_dlsym();
    if (cuda) {
        q_sid[0] = (sid_t)real_dlsym(cuda, "cuStreamGetId");
        q_sid[1] = (sid_t)real_dlsym(cuda, "cuStreamGetId_ptsz");
        // The first version (status and id), whatever cuda.h maps the name to now.
        q_scap[0] = (scap_t)real_dlsym(cuda, "cuStreamGetCaptureInfo");
        q_scap[1] = (scap_t)real_dlsym(cuda, "cuStreamGetCaptureInfo_ptsz");
        q_sdev[0] = (sdev_t)real_dlsym(cuda, "cuStreamGetDevice");
        q_sdev[1] = (sdev_t)real_dlsym(cuda, "cuStreamGetDevice_ptsz");
        q_sattr[0] = (sattr_t)real_dlsym(cuda, "cuStreamGetAttribute");
        q_sattr[1] = (sattr_t)real_dlsym(cuda, "cuStreamGetAttribute_ptsz");
    }
    __atomic_store_n(&q_init, 1, __ATOMIC_RELEASE);
}
struct sinfo {
    long long id;   // -1: unknown
    int dev;        // the stream's device (else the current one)
    int capturing;  // 0, 1, or -1 (unknown)
    unsigned long long cap_id;
};
static struct sinfo stream_info(CUstream s, int ptsz) {
    stream_queries();
    struct sinfo r = {-1, -1, -1, 0};
    CUstreamCaptureStatus st;
    cuuint64_t cid = 0;
    if (q_scap[ptsz] && q_scap[ptsz](s, &st, &cid) == CUDA_SUCCESS) {
        // An invalidated capture is still a capture: its work does not run now either.
        r.capturing = st != CU_STREAM_CAPTURE_STATUS_NONE;
        r.cap_id = r.capturing ? (unsigned long long)cid : 0;
    }
    r.dev = cur_device();
    if (r.capturing == 0) {
        unsigned long long id;
        if (q_sid[ptsz] && q_sid[ptsz](s, &id) == CUDA_SUCCESS) r.id = (long long)id;
        CUdevice d;
        if (q_sdev[ptsz] && q_sdev[ptsz](s, &d) == CUDA_SUCCESS) r.dev = (int)d;
    }
    return r;
}
// `"stream_id": .., "ptsz": .., "capturing": .., "capture_id": .., ` (no stream_id when the
// query failed).
static void put_stream_info(FILE *fp, struct sinfo si, int ptsz) {
    if (si.id >= 0) fprintf(fp, "\"stream_id\": %lld, ", si.id);
    fprintf(fp, "\"ptsz\": %d, \"capturing\": %d, \"capture_id\": %llu, ", ptsz, si.capturing,
            si.cap_id);
}
// The physical memory synchronization domain of the launch: its logical domain (default or
// remote; the launch's attribute, else the stream's) through the domain map (the launch's,
// else the stream's; by default default -> 0, remote -> 1). Kernels in different domains are
// "system-scope distanced" (cuda.h, CUlaunchAttributeValue::memSyncDomain): S-SCOPE.
static int mem_domain(CUstream s, int ptsz) {
    stream_queries();
    int dom = cur_msd, m0 = cur_msdmap0, m1 = cur_msdmap1;
    CUlaunchAttributeValue v;
    if (dom < 0) {
        dom = 0;
        memset(&v, 0, sizeof v);
        if (q_sattr[ptsz] && q_sattr[ptsz](s, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, &v) == CUDA_SUCCESS)
            dom = (int)v.memSyncDomain;
    }
    if (m0 < 0) {
        m0 = 0;
        m1 = 1;
        memset(&v, 0, sizeof v);
        if (q_sattr[ptsz] && q_sattr[ptsz](s, CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &v) == CUDA_SUCCESS) {
            m0 = v.memSyncDomainMap.default_;
            m1 = v.memSyncDomainMap.remote;
        }
    }
    return dom == CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE ? m1 : m0;
}
static void reset_order_attrs(void) {
    cur_pev = cur_cev = 0;
    cur_pev_start = 0;
    cur_msd = cur_msdmap0 = cur_msdmap1 = -1;
}

// The record of a launch, built in memory by dump_k and written by launch_done once the
// launch returned (per thread).
static __thread char *rec_buf;
static __thread size_t rec_len;

static void dump_k(CUkernel k, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                 unsigned by, unsigned bz, unsigned smem, unsigned cx, unsigned cy,
                 unsigned cz, void **params) {
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    getparam_t info = (getparam_t)dlsym(cuda, "cuFuncGetParamInfo");
    kgetparam_t kinfo = (kgetparam_t)dlsym(cuda, "cuKernelGetParamInfo");
    getname_t name = (getname_t)dlsym(cuda, "cuFuncGetName");
    free(rec_buf);
    rec_buf = 0;
    FILE *fp = open_memstream(&rec_buf, &rec_len);
    if (!fp) return;
    const char *nm = "?";
    if (name && f && name(&nm, f) != CUDA_SUCCESS) nm = "?";
    // The runtime may hand a CUkernel (library-managed, lazily loaded) where a CUfunction is
    // expected; then the CUfunction queries fail and the CUkernel ones work.
    if (f && !k) {
        size_t o, z;
        if (info && info(f, 0, &o, &z) != CUDA_SUCCESS && kinfo && kinfo((CUkernel)f, 0, &o, &z) == CUDA_SUCCESS)
            k = (CUkernel)f;
    }
    if (k) {
        typedef CUresult (*kname_t)(const char **, CUkernel);
        kname_t kn = (kname_t)dlsym(cuda, "cuKernelGetName");
        if (kn) kn(&nm, k);
    }
    // The launched kernel's module (CUfunction) or library (CUkernel) handle: the process can
    // look up the module's data globals in it (cuModuleGetGlobal / cuLibraryGetGlobal,
    // tools/capture_dist.py).
    unsigned long long hmod = 0, hlib = 0;
    {
        typedef CUresult (*fmod_t)(CUmodule *, CUfunction);
        typedef CUresult (*klib_t)(CUlibrary *, CUkernel);
        fmod_t fmod = (fmod_t)dlsym(cuda, "cuFuncGetModule");
        klib_t klib = (klib_t)dlsym(cuda, "cuKernelGetLibrary");
        CUmodule md = 0;
        CUlibrary lb = 0;
        if (k && klib && klib(&lb, k) == CUDA_SUCCESS) hlib = (unsigned long long)lb;
        else if (f && fmod && fmod(&md, f) == CUDA_SUCCESS) hmod = (unsigned long long)md;
    }
    struct sinfo si = stream_info((CUstream)cur_stream, cur_ptsz);
    // No snapshot of a launch into a capturing stream: it does not run now, and synchronizing
    // the context would invalidate the capture.
    snap_on = snap_allowed && snap_match(nm) && si.capturing == 0;
    snap_len = 0;
    if (snap_on && !snap_buf && !(snap_buf = malloc(SNAP_MAX))) snap_on = 0;
    if (snap_on) {
        snap_cur = __atomic_fetch_add(&snap_seq, 1, __ATOMIC_RELAXED);
        fprintf(fp, "{\"snap\": %u, ", snap_cur);
    }
    fprintf(fp, "%s\"name\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
                "\"smem\": %u, \"cluster\": [%u, %u, %u], \"pdl\": %d, \"stream\": %llu, "
                "\"dev\": %d, \"mod\": %llu, \"lib\": %llu, ",
            snap_on ? "" : "{", nm, gx, gy, gz, bx, by, bz, smem, cx, cy, cz, cur_pdl, cur_stream, cur_device(),
            hmod, hlib);
    put_stream_info(fp, si, cur_ptsz);
    fprintf(fp, "\"mem_sync_domain\": %d, ", mem_domain((CUstream)cur_stream, cur_ptsz));
    if (cur_pev)
        fprintf(fp, "\"prog_event\": {\"event\": %llu, \"trigger_at_block_start\": %d}, ", cur_pev,
                cur_pev_start);
    if (cur_cev) fprintf(fp, "\"completion_event\": %llu, ", cur_cev);
    if (cur_pref[0])
        fprintf(fp, "\"preferred_cluster\": [%u, %u, %u], ", cur_pref[0], cur_pref[1], cur_pref[2]);
    fprintf(fp, "\"params\": [");
    if (!params && cur_extra) {
        const unsigned char *buf = 0;
        size_t bsize = 0;
        for (void **e = cur_extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
            if (e[0] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = (const unsigned char *)e[1];
            if (e[0] == CU_LAUNCH_PARAM_BUFFER_SIZE) bsize = *(size_t *)e[1];
        }
        if (buf) {
            fprintf(fp, "{\"offset\": 0, \"packed\": true, \"hex\": \"");
            for (size_t j = 0; j < bsize; j++) fprintf(fp, "%02x", buf[j]);
            snap_param(0, buf, bsize);
            fprintf(fp, "\"}");
        }
    }
    for (size_t i = 0; params; i++) {
        size_t off, size;
        CUresult r = k ? (kinfo ? kinfo(k, i, &off, &size) : CUDA_ERROR_NOT_SUPPORTED)
                       : (info ? info(f, i, &off, &size) : CUDA_ERROR_NOT_SUPPORTED);
        if (r != CUDA_SUCCESS) break;
        fprintf(fp, "%s{\"offset\": %zu, \"hex\": \"", i ? ", " : "", off);
        const unsigned char *p = (const unsigned char *)params[i];
        for (size_t j = 0; j < size; j++) fprintf(fp, "%02x", p[j]);
        snap_param(off, p, size);
        fprintf(fp, "\"}");
    }
    fprintf(fp, "]");
    fclose(fp);
}

#define dump(f, ...) dump_k(0, f, __VA_ARGS__)

// Write the launch record dump_k built, if the launch succeeded (a failed launch leaves no
// record), with "t" = "t0" (host time just before the call: the order in which ranks
// submitted their work), "t1" (after it returned) and the thread.
static void launch_done(int ok, unsigned long long t0, unsigned long long t1) {
    if (rec_buf && ok) {
        FILE *fp = out_begin();
        if (fp) {
            fprintf(fp, "%s, \"t\": %llu, \"t0\": %llu, \"t1\": %llu, \"tid\": %ld}\n", rec_buf, t0,
                    t0, t1, tid());
            out_end(fp);
        }
    }
    free(rec_buf);
    rec_buf = 0;
    rec_len = 0;
}

static void snap_pre(void) {
    if (snap_on) snap_dump("pre");
}
static void snap_post(void) {
    if (!snap_on) return;
    snap_dump("post");
    snap_on = 0;
}

typedef CUresult (*launch_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, CUstream, void **, void **);
typedef CUresult (*launchex_t)(const CUlaunchConfig *, CUfunction, void **, void **);
// The real entry points, legacy ([0]) and per-thread default stream ([1]) flavour: a process
// may use both (stream 0 means a different stream in each).
static launch_t real_launch[2];
static launchex_t real_launchex[2];

// The launch-order attributes of a launch (driver and runtime structs).
static void order_attr(const CUlaunchAttribute *a) {
    switch (a->id) {
    case CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT:
        cur_pev = (unsigned long long)a->value.programmaticEvent.event;
        cur_pev_start = a->value.programmaticEvent.triggerAtBlockStart != 0;
        break;
    case CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT:
        cur_cev = (unsigned long long)a->value.launchCompletionEvent.event;
        break;
    case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN:
        cur_msd = (int)a->value.memSyncDomain;
        break;
    case CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP:
        cur_msdmap0 = a->value.memSyncDomainMap.default_;
        cur_msdmap1 = a->value.memSyncDomainMap.remote;
        break;
    default:
        break;
    }
}
static void order_attr_rt(const cudaLaunchAttribute *a) {
    switch (a->id) {
    case cudaLaunchAttributeProgrammaticEvent:
        cur_pev = (unsigned long long)a->val.programmaticEvent.event;
        cur_pev_start = a->val.programmaticEvent.triggerAtBlockStart != 0;
        break;
    case cudaLaunchAttributeLaunchCompletionEvent:
        cur_cev = (unsigned long long)a->val.launchCompletionEvent.event;
        break;
    case cudaLaunchAttributeMemSyncDomain:
        cur_msd = (int)a->val.memSyncDomain;
        break;
    case cudaLaunchAttributeMemSyncDomainMap:
        cur_msdmap0 = a->val.memSyncDomainMap.default_;
        cur_msdmap1 = a->val.memSyncDomainMap.remote;
        break;
    default:
        break;
    }
}

// The next definition of `sym` after this library (glibc's dlsym directly: this library's own
// dlsym would hand back its hook).
static void *next_sym(const char *sym) {
    init_real_dlsym();
    return real_dlsym(RTLD_NEXT, sym);
}

static CUresult launch_hook(int ptsz, CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz, unsigned smem, CUstream s,
                            void **params, void **extra) {
    if (!real_launch[ptsz])
        real_launch[ptsz] = (launch_t)next_sym(ptsz ? "cuLaunchKernel_ptsz" : "cuLaunchKernel");
    launch_t real = real_launch[ptsz];
    if (in_rt) return real(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
    cur_extra = extra;
    cur_stream = (unsigned long long)s;
    cur_ptsz = ptsz;
    snap_allowed = 1;
    dump(f, gx, gy, gz, bx, by, bz, smem, 1, 1, 1, params);
    snap_allowed = 0;
    cur_extra = 0;
    cur_ptsz = 0;
    snap_pre();
    unsigned long long t0 = now_ns();
    CUresult r = real(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
    unsigned long long t1 = now_ns();
    snap_post();
    launch_done(r == CUDA_SUCCESS, t0, t1);
    return r;
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                        unsigned by, unsigned bz, unsigned smem, CUstream s, void **params,
                        void **extra) {
    return launch_hook(0, f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
}
CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                             unsigned by, unsigned bz, unsigned smem, CUstream s, void **params,
                             void **extra) {
    return launch_hook(1, f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
}

static CUresult launchex_hook(int ptsz, const CUlaunchConfig *c, CUfunction f, void **params,
                              void **extra) {
    if (!real_launchex[ptsz])
        real_launchex[ptsz] = (launchex_t)next_sym(ptsz ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx");
    launchex_t real = real_launchex[ptsz];
    if (in_rt) return real(c, f, params, extra);
    unsigned cx = 1, cy = 1, cz = 1;
    for (unsigned i = 0; i < c->numAttrs; i++)
        if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
            cx = c->attrs[i].value.clusterDim.x;
            cy = c->attrs[i].value.clusterDim.y;
            cz = c->attrs[i].value.clusterDim.z;
        } else if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
            cur_pdl = c->attrs[i].value.programmaticStreamSerializationAllowed != 0;
        } else if (c->attrs[i].id == CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION) {
            cur_pref[0] = c->attrs[i].value.preferredClusterDim.x;
            cur_pref[1] = c->attrs[i].value.preferredClusterDim.y;
            cur_pref[2] = c->attrs[i].value.preferredClusterDim.z;
        } else {
            order_attr(&c->attrs[i]);
        }
    cur_extra = extra;
    cur_stream = (unsigned long long)c->hStream;
    cur_ptsz = ptsz;
    snap_allowed = 1;
    dump(f, c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY, c->blockDimZ,
         c->sharedMemBytes, cx, cy, cz, params);
    snap_allowed = 0;
    cur_extra = 0;
    cur_pdl = 0;
    cur_ptsz = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    reset_order_attrs();
    snap_pre();
    unsigned long long t0 = now_ns();
    CUresult r = real(c, f, params, extra);
    unsigned long long t1 = now_ns();
    snap_post();
    launch_done(r == CUDA_SUCCESS, t0, t1);
    return r;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **params, void **extra) {
    return launchex_hook(0, c, f, params, extra);
}
CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *c, CUfunction f, void **params, void **extra) {
    return launchex_hook(1, c, f, params, extra);
}

// Code images (LAUNCHCAP_IMAGES=dir): every image handed to a module / library loader is
// written to dir/<n>.bin, so a library kernel's cubin can be found by its name afterwards.
// An image is an ELF, a fatbin (header: magic, version, header size, payload size) or a
// fatbin wrapper (magic, version, pointer to the fatbin).
static size_t image_size(const unsigned char *p) {
    unsigned magic = *(const unsigned *)p;
    if (magic == 0x464c457f) {  // ELF64: the furthest of the header tables and sections
        unsigned long long phoff = *(const unsigned long long *)(p + 0x20);
        unsigned long long shoff = *(const unsigned long long *)(p + 0x28);
        unsigned short phentsize = *(const unsigned short *)(p + 0x36);
        unsigned short phnum = *(const unsigned short *)(p + 0x38);
        unsigned short shentsize = *(const unsigned short *)(p + 0x3a);
        unsigned short shnum = *(const unsigned short *)(p + 0x3c);
        size_t end = shoff + (size_t)shentsize * shnum;
        if (phoff + (size_t)phentsize * phnum > end) end = phoff + (size_t)phentsize * phnum;
        for (unsigned i = 0; i < shnum; i++) {
            const unsigned char *sh = p + shoff + (size_t)i * shentsize;
            unsigned type = *(const unsigned *)(sh + 4);
            unsigned long long off = *(const unsigned long long *)(sh + 0x18);
            unsigned long long size = *(const unsigned long long *)(sh + 0x20);
            if (type != 8 && off + size > end) end = off + size;  // not SHT_NOBITS
        }
        return end;
    }
    if (magic == 0xba55ed50)
        return *(const unsigned short *)(p + 6) + *(const unsigned long long *)(p + 8);
    return 0;
}

static void save_image(const void *img) {
    const char *dir = getenv("LAUNCHCAP_IMAGES");
    if (!dir || !img) return;
    const unsigned char *p = img;
    if (*(const unsigned *)p == 0x466243b1) p = *(const unsigned char *const *)(p + 8);
    size_t n = image_size(p);
    if (!n) return;
    static int count;
    char path[4096], dbuf[4096];
    dir = rank_path("LAUNCHCAP_IMAGES", dir, dbuf, sizeof dbuf);
    snprintf(path, sizeof path, "%s/%d.bin", dir, count++);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(p, 1, n, f);
    fclose(f);
}

typedef CUresult (*mld_t)(CUmodule *, const void *);
typedef CUresult (*mldex_t)(CUmodule *, const void *, unsigned, CUjit_option *, void **);
typedef CUresult (*lld_t)(CUlibrary *, const void *, CUjit_option *, void **, unsigned,
                          CUlibraryOption *, void **, unsigned);
static mld_t real_mld, real_mfat;
static mldex_t real_mldex;
static lld_t real_lld;

static CUresult hook_mld(CUmodule *m, const void *img) {
    save_image(img);
    return real_mld(m, img);
}
static CUresult hook_mfat(CUmodule *m, const void *img) {
    save_image(img);
    return real_mfat(m, img);
}
static CUresult hook_mldex(CUmodule *m, const void *img, unsigned n, CUjit_option *o, void **v) {
    save_image(img);
    return real_mldex(m, img, n, o, v);
}
static CUresult hook_lld(CUlibrary *l, const void *code, CUjit_option *jo, void **jv, unsigned jn,
                         CUlibraryOption *lo, void **lv, unsigned ln) {
    save_image(code);
    return real_lld(l, code, jo, jv, jn, lo, lv, ln);
}

// Swap a loader entry point for its hook (by name), remembering the real one.
static void *wrap_loader(const char *sym, void *p) {
    if (!strcmp(sym, "cuModuleLoadData")) { real_mld = (mld_t)p; return (void *)hook_mld; }
    if (!strcmp(sym, "cuModuleLoadFatBinary")) { real_mfat = (mld_t)p; return (void *)hook_mfat; }
    if (!strcmp(sym, "cuModuleLoadDataEx")) { real_mldex = (mldex_t)p; return (void *)hook_mldex; }
    if (!strcmp(sym, "cuLibraryLoadData")) { real_lld = (lld_t)p; return (void *)hook_lld; }
    return p;
}

// Virtual memory management, IPC, multicast and stream memory calls (LAUNCHCAP_VMM=1),
// logged in order with the launches as {"vmm": ...} / {"ev": ...} lines. A multi-process
// capture (tools/capture_dist.py) uses them to resolve a pointer into a peer's memory: torch's
// symmetric memory exports each rank's cuMemCreate allocation and maps it into every peer
// (cuMemImportFromShareableHandle + cuMemMap), and binds it to a multicast object (NVLS).
static void vlog(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#include <stdarg.h>
static void vlog(const char *fmt, ...) {
    FILE *fp = out_begin();
    if (!fp) return;
    va_list a;
    va_start(a, fmt);
    vfprintf(fp, fmt, a);
    va_end(a);
    fprintf(fp, "\n");
    out_end(fp);
}
static void hex_into(char *dst, const void *src, size_t n) {
    const unsigned char *p = src;
    for (size_t i = 0; i < n; i++) sprintf(dst + 2 * i, "%02x", p[i]);
    dst[2 * n] = 0;
}
// Identity of an exported / imported file descriptor that survives SCM_RIGHTS: the open file
// itself (device and inode as fstat reports them).
static void fd_id(int fd, unsigned long long *dev, unsigned long long *ino) {
    struct stat st;
    *dev = *ino = 0;
    if (fd >= 0 && fstat(fd, &st) == 0) {
        *dev = (unsigned long long)st.st_dev;
        *ino = (unsigned long long)st.st_ino;
    }
}

typedef CUresult (*memcreate_t)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *,
                                unsigned long long);
typedef CUresult (*memexport_t)(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType,
                                unsigned long long);
typedef CUresult (*memimport_t)(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
typedef CUresult (*memreserve_t)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
typedef CUresult (*memmap_t)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                             unsigned long long);
typedef CUresult (*memunmap_t)(CUdeviceptr, size_t);
typedef CUresult (*memrelease_t)(CUmemGenericAllocationHandle);
typedef CUresult (*mccreate_t)(CUmemGenericAllocationHandle *, const CUmulticastObjectProp *);
typedef CUresult (*mcadd_t)(CUmemGenericAllocationHandle, CUdevice);
typedef CUresult (*mcbind_t)(CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle,
                             size_t, size_t, unsigned long long);
typedef CUresult (*mcbindaddr_t)(CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t,
                                 unsigned long long);
typedef CUresult (*mcunbind_t)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
typedef CUresult (*ipcget_t)(CUipcMemHandle *, CUdeviceptr);
typedef CUresult (*ipcopen_t)(CUdeviceptr *, CUipcMemHandle, unsigned);
typedef CUresult (*wv32_t)(CUstream, CUdeviceptr, cuuint32_t, unsigned);
typedef CUresult (*memsetd32a_t)(CUdeviceptr, unsigned, size_t, CUstream);

static memcreate_t r_memcreate;
static memexport_t r_memexport;
static memimport_t r_memimport;
static memreserve_t r_memreserve;
static memmap_t r_memmap;
static memunmap_t r_memunmap;
static memrelease_t r_memrelease;
static mccreate_t r_mccreate;
static mcadd_t r_mcadd;
static mcbind_t r_mcbind;
static mcbindaddr_t r_mcbindaddr;
static mcunbind_t r_mcunbind;
static ipcget_t r_ipcget;
static ipcopen_t r_ipcopen;
static wv32_t r_wv32;
static memsetd32a_t r_memsetd32a;

static CUresult h_memcreate(CUmemGenericAllocationHandle *h, size_t size,
                            const CUmemAllocationProp *p, unsigned long long fl) {
    CUresult r = r_memcreate(h, size, p, fl);
    vlog("{\"vmm\": \"create\", \"r\": %d, \"handle\": %llu, \"size\": %zu, \"dev\": %d, "
         "\"loc_type\": %d, \"handle_types\": %d, \"t\": %llu}",
         r, r ? 0ull : (unsigned long long)*h, size, p ? p->location.id : -1,
         p ? (int)p->location.type : -1, p ? (int)p->requestedHandleTypes : -1, now_ns());
    return r;
}
static CUresult h_memexport(void *sh, CUmemGenericAllocationHandle h, CUmemAllocationHandleType t,
                            unsigned long long fl) {
    CUresult r = r_memexport(sh, h, t, fl);
    char hx[2 * 64 + 1] = "";
    unsigned long long dev = 0, ino = 0;
    int fd = -1;
    if (!r && t == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        fd = *(int *)sh;
        fd_id(fd, &dev, &ino);
    } else if (!r && t == CU_MEM_HANDLE_TYPE_FABRIC) {
        hex_into(hx, sh, 64);
    }
    vlog("{\"vmm\": \"export\", \"r\": %d, \"handle\": %llu, \"type\": %d, \"fd\": %d, "
         "\"fd_dev\": %llu, \"fd_ino\": %llu, \"fabric\": \"%s\", \"pid\": %d, \"t\": %llu}",
         r, (unsigned long long)h, (int)t, fd, dev, ino, hx, (int)getpid(), now_ns());
    return r;
}
static CUresult h_memimport(CUmemGenericAllocationHandle *h, void *os, CUmemAllocationHandleType t) {
    char hx[2 * 64 + 1] = "";
    unsigned long long dev = 0, ino = 0;
    int fd = -1;
    if (t == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        fd = (int)(long)os;
        fd_id(fd, &dev, &ino);
    } else if (t == CU_MEM_HANDLE_TYPE_FABRIC && os) {
        hex_into(hx, os, 64);
    }
    CUresult r = r_memimport(h, os, t);
    vlog("{\"vmm\": \"import\", \"r\": %d, \"handle\": %llu, \"type\": %d, \"fd\": %d, "
         "\"fd_dev\": %llu, \"fd_ino\": %llu, \"fabric\": \"%s\", \"t\": %llu}",
         r, r ? 0ull : (unsigned long long)*h, (int)t, fd, dev, ino, hx, now_ns());
    return r;
}
static CUresult h_memreserve(CUdeviceptr *p, size_t size, size_t al, CUdeviceptr addr,
                             unsigned long long fl) {
    CUresult r = r_memreserve(p, size, al, addr, fl);
    vlog("{\"vmm\": \"reserve\", \"r\": %d, \"ptr\": %llu, \"size\": %zu, \"t\": %llu}", r,
         r ? 0ull : (unsigned long long)*p, size, now_ns());
    return r;
}
static CUresult h_memmap(CUdeviceptr p, size_t size, size_t off, CUmemGenericAllocationHandle h,
                         unsigned long long fl) {
    CUresult r = r_memmap(p, size, off, h, fl);
    vlog("{\"vmm\": \"map\", \"r\": %d, \"ptr\": %llu, \"size\": %zu, \"offset\": %zu, "
         "\"handle\": %llu, \"t\": %llu}",
         r, (unsigned long long)p, size, off, (unsigned long long)h, now_ns());
    return r;
}
static CUresult h_memunmap(CUdeviceptr p, size_t size) {
    vlog("{\"vmm\": \"unmap\", \"ptr\": %llu, \"size\": %zu, \"t\": %llu}", (unsigned long long)p,
         size, now_ns());
    return r_memunmap(p, size);
}
static CUresult h_memrelease(CUmemGenericAllocationHandle h) {
    vlog("{\"vmm\": \"release\", \"handle\": %llu, \"t\": %llu}", (unsigned long long)h, now_ns());
    return r_memrelease(h);
}
static CUresult h_mccreate(CUmemGenericAllocationHandle *h, const CUmulticastObjectProp *p) {
    CUresult r = r_mccreate(h, p);
    vlog("{\"vmm\": \"mc_create\", \"r\": %d, \"handle\": %llu, \"size\": %zu, \"ndev\": %u, "
         "\"handle_types\": %d, \"t\": %llu}",
         r, r ? 0ull : (unsigned long long)*h, p ? p->size : 0, p ? p->numDevices : 0,
         p ? (int)p->handleTypes : -1, now_ns());
    return r;
}
static CUresult h_mcadd(CUmemGenericAllocationHandle h, CUdevice d) {
    CUresult r = r_mcadd(h, d);
    vlog("{\"vmm\": \"mc_add_device\", \"r\": %d, \"handle\": %llu, \"dev\": %d}", r,
         (unsigned long long)h, (int)d);
    return r;
}
static CUresult h_mcbind(CUmemGenericAllocationHandle mc, size_t mcoff,
                         CUmemGenericAllocationHandle mem, size_t memoff, size_t size,
                         unsigned long long fl) {
    CUresult r = r_mcbind(mc, mcoff, mem, memoff, size, fl);
    vlog("{\"vmm\": \"mc_bind_mem\", \"r\": %d, \"mc\": %llu, \"mc_offset\": %zu, \"handle\": "
         "%llu, \"offset\": %zu, \"size\": %zu}",
         r, (unsigned long long)mc, mcoff, (unsigned long long)mem, memoff, size);
    return r;
}
// cuMulticastBindAddr binds by virtual address (the capture resolves it through the rank's
// mappings); cuMulticastUnbind ends a bind.
static CUresult h_mcbindaddr(CUmemGenericAllocationHandle mc, size_t mcoff, CUdeviceptr p,
                             size_t size, unsigned long long fl) {
    CUresult r = r_mcbindaddr(mc, mcoff, p, size, fl);
    vlog("{\"vmm\": \"mc_bind_addr\", \"r\": %d, \"mc\": %llu, \"mc_offset\": %zu, \"ptr\": "
         "%llu, \"size\": %zu}",
         r, (unsigned long long)mc, mcoff, (unsigned long long)p, size);
    return r;
}
static CUresult h_mcunbind(CUmemGenericAllocationHandle mc, CUdevice d, size_t mcoff, size_t size) {
    CUresult r = r_mcunbind(mc, d, mcoff, size);
    vlog("{\"vmm\": \"mc_unbind\", \"r\": %d, \"mc\": %llu, \"dev\": %d, \"mc_offset\": %zu, "
         "\"size\": %zu}",
         r, (unsigned long long)mc, (int)d, mcoff, size);
    return r;
}
static CUresult h_ipcget(CUipcMemHandle *h, CUdeviceptr p) {
    CUresult r = r_ipcget(h, p);
    char hx[2 * 64 + 1] = "";
    if (!r) hex_into(hx, h->reserved, 64);
    vlog("{\"vmm\": \"ipc_get\", \"r\": %d, \"ptr\": %llu, \"ipc\": \"%s\"}", r,
         (unsigned long long)p, hx);
    return r;
}
static CUresult h_ipcopen(CUdeviceptr *p, CUipcMemHandle h, unsigned fl) {
    CUresult r = r_ipcopen(p, h, fl);
    char hx[2 * 64 + 1];
    hex_into(hx, h.reserved, 64);
    vlog("{\"vmm\": \"ipc_open\", \"r\": %d, \"ptr\": %llu, \"ipc\": \"%s\"}", r,
         r ? 0ull : (unsigned long long)*p, hx);
    return r;
}
static CUresult h_wv32(CUstream s, CUdeviceptr p, cuuint32_t v, unsigned fl) {
    vlog("{\"ev\": \"write_value32\", \"stream\": %llu, \"ptr\": %llu, \"value\": %u, \"t\": %llu}",
         (unsigned long long)s, (unsigned long long)p, v, now_ns());
    return r_wv32(s, p, v, fl);
}
static CUresult h_memsetd32a(CUdeviceptr p, unsigned v, size_t n, CUstream s) {
    vlog("{\"op\": \"memset32\", \"dst\": %llu, \"src\": 0, \"value\": %u, \"bytes\": %zu, "
         "\"kind\": 0, \"stream\": %llu}",
         (unsigned long long)p, v, 4 * n, (unsigned long long)s);
    return r_memsetd32a(p, v, n, s);
}

typedef CUresult (*memalloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*memfree_t)(CUdeviceptr);
typedef CUresult (*hostalloc_t)(void **, size_t, unsigned);
typedef CUresult (*allochost_t)(void **, size_t);
typedef CUresult (*hostdevptr_t)(CUdeviceptr *, void *, unsigned);
static memalloc_t r_memalloc;
static memfree_t r_memfree;
static hostalloc_t r_hostalloc;
static allochost_t r_allochost;
static hostdevptr_t r_hostdevptr;
// Plain device allocations (cudaMalloc: DeepEP's NVLink buffer and its pointer tables, which
// the caching allocator does not see) and mapped host memory (DeepEP's host-visible counters).
static CUresult h_memalloc(CUdeviceptr *p, size_t n) {
    CUresult r = r_memalloc(p, n);
    vlog("{\"vmm\": \"alloc\", \"r\": %d, \"ptr\": %llu, \"size\": %zu, \"t\": %llu}", r,
         r ? 0ull : (unsigned long long)*p, n, now_ns());
    return r;
}
static CUresult h_memfree(CUdeviceptr p) {
    vlog("{\"vmm\": \"free\", \"ptr\": %llu, \"t\": %llu}", (unsigned long long)p, now_ns());
    return r_memfree(p);
}
static CUresult h_hostalloc(void **p, size_t n, unsigned fl) {
    CUresult r = r_hostalloc(p, n, fl);
    vlog("{\"vmm\": \"host_alloc\", \"r\": %d, \"ptr\": %llu, \"size\": %zu, \"flags\": %u}", r,
         r ? 0ull : (unsigned long long)*p, n, fl);
    return r;
}
static CUresult h_allochost(void **p, size_t n) {
    CUresult r = r_allochost(p, n);
    vlog("{\"vmm\": \"host_alloc\", \"r\": %d, \"ptr\": %llu, \"size\": %zu, \"flags\": 0}", r,
         r ? 0ull : (unsigned long long)*p, n);
    return r;
}
static CUresult h_hostdevptr(CUdeviceptr *d, void *p, unsigned fl) {
    CUresult r = r_hostdevptr(d, p, fl);
    vlog("{\"vmm\": \"host_devptr\", \"r\": %d, \"host\": %llu, \"ptr\": %llu}", r,
         (unsigned long long)p, r ? 0ull : (unsigned long long)*d);
    return r;
}

#define VMM_HOOK(name, real, hook)                       \
    if (!strcmp(sym, name)) {                            \
        if ((void *)hook == p) return p;                 \
        real = (__typeof__(real))p;                      \
        return (void *)hook;                             \
    }
static void *wrap_vmm(const char *sym, void *p);
static void *wrap_vmm_impl(const char *sym, void *p) {
    VMM_HOOK("cuMemCreate", r_memcreate, h_memcreate)
    VMM_HOOK("cuMemExportToShareableHandle", r_memexport, h_memexport)
    VMM_HOOK("cuMemImportFromShareableHandle", r_memimport, h_memimport)
    VMM_HOOK("cuMemAddressReserve", r_memreserve, h_memreserve)
    VMM_HOOK("cuMemMap", r_memmap, h_memmap)
    VMM_HOOK("cuMemUnmap", r_memunmap, h_memunmap)
    VMM_HOOK("cuMemRelease", r_memrelease, h_memrelease)
    VMM_HOOK("cuMulticastCreate", r_mccreate, h_mccreate)
    VMM_HOOK("cuMulticastAddDevice", r_mcadd, h_mcadd)
    VMM_HOOK("cuMulticastBindMem", r_mcbind, h_mcbind)
    VMM_HOOK("cuMulticastBindAddr", r_mcbindaddr, h_mcbindaddr)
    VMM_HOOK("cuMulticastUnbind", r_mcunbind, h_mcunbind)
    VMM_HOOK("cuIpcGetMemHandle", r_ipcget, h_ipcget)
    VMM_HOOK("cuIpcOpenMemHandle", r_ipcopen, h_ipcopen)
    VMM_HOOK("cuStreamWriteValue32", r_wv32, h_wv32)
    VMM_HOOK("cuMemsetD32Async", r_memsetd32a, h_memsetd32a)
    VMM_HOOK("cuMemAlloc", r_memalloc, h_memalloc)
    VMM_HOOK("cuMemFree", r_memfree, h_memfree)
    VMM_HOOK("cuMemHostAlloc", r_hostalloc, h_hostalloc)
    VMM_HOOK("cuMemAllocHost", r_allochost, h_allochost)
    VMM_HOOK("cuMemHostGetDevicePointer", r_hostdevptr, h_hostdevptr)
    return p;
}

// Libraries that link libcuda's VMM entry points directly (mscclpp's _mscclpp.so: cuMemCreate,
// cuMemMap, ... through the PLT) bypass dlsym / cuGetProcAddress: interpose the exported names.
#define VMM_EXPORT(ret_t, name, real, hook, params, args)                         \
    ret_t name params {                                                          \
        if (!real) real = (__typeof__(real))dlsym(RTLD_NEXT, #name);             \
        if (!getenv("LAUNCHCAP_VMM")) return real args;                          \
        return hook args;                                                        \
    }
VMM_EXPORT(CUresult, cuMemCreate, r_memcreate, h_memcreate,
           (CUmemGenericAllocationHandle *h, size_t size, const CUmemAllocationProp *p,
            unsigned long long fl),
           (h, size, p, fl))
VMM_EXPORT(CUresult, cuMemExportToShareableHandle, r_memexport, h_memexport,
           (void *sh, CUmemGenericAllocationHandle h, CUmemAllocationHandleType t,
            unsigned long long fl),
           (sh, h, t, fl))
VMM_EXPORT(CUresult, cuMemImportFromShareableHandle, r_memimport, h_memimport,
           (CUmemGenericAllocationHandle *h, void *os, CUmemAllocationHandleType t), (h, os, t))
VMM_EXPORT(CUresult, cuMemAddressReserve, r_memreserve, h_memreserve,
           (CUdeviceptr *p, size_t size, size_t al, CUdeviceptr addr, unsigned long long fl),
           (p, size, al, addr, fl))
VMM_EXPORT(CUresult, cuMemMap, r_memmap, h_memmap,
           (CUdeviceptr p, size_t size, size_t off, CUmemGenericAllocationHandle h,
            unsigned long long fl),
           (p, size, off, h, fl))
VMM_EXPORT(CUresult, cuMemUnmap, r_memunmap, h_memunmap, (CUdeviceptr p, size_t size), (p, size))
VMM_EXPORT(CUresult, cuMemRelease, r_memrelease, h_memrelease, (CUmemGenericAllocationHandle h),
           (h))
VMM_EXPORT(CUresult, cuMulticastCreate, r_mccreate, h_mccreate,
           (CUmemGenericAllocationHandle *h, const CUmulticastObjectProp *p), (h, p))
VMM_EXPORT(CUresult, cuMulticastAddDevice, r_mcadd, h_mcadd,
           (CUmemGenericAllocationHandle h, CUdevice d), (h, d))
VMM_EXPORT(CUresult, cuMulticastBindMem, r_mcbind, h_mcbind,
           (CUmemGenericAllocationHandle mc, size_t mcoff, CUmemGenericAllocationHandle mem,
            size_t memoff, size_t size, unsigned long long fl),
           (mc, mcoff, mem, memoff, size, fl))
VMM_EXPORT(CUresult, cuMulticastBindAddr, r_mcbindaddr, h_mcbindaddr,
           (CUmemGenericAllocationHandle mc, size_t mcoff, CUdeviceptr p, size_t size,
            unsigned long long fl),
           (mc, mcoff, p, size, fl))
VMM_EXPORT(CUresult, cuMulticastUnbind, r_mcunbind, h_mcunbind,
           (CUmemGenericAllocationHandle mc, CUdevice d, size_t mcoff, size_t size),
           (mc, d, mcoff, size))

static void *wrap_vmm(const char *sym, void *p) {
    if (p == (void *)cuMemCreate || p == (void *)cuMemExportToShareableHandle ||
        p == (void *)cuMemImportFromShareableHandle || p == (void *)cuMemAddressReserve ||
        p == (void *)cuMemMap || p == (void *)cuMemUnmap || p == (void *)cuMemRelease ||
        p == (void *)cuMulticastCreate || p == (void *)cuMulticastAddDevice ||
        p == (void *)cuMulticastBindMem || p == (void *)cuMulticastBindAddr ||
        p == (void *)cuMulticastUnbind)
        return p;
    return wrap_vmm_impl(sym, p);
}

// Stream order (LAUNCHCAP_ORDER=1; LAUNCHCAP_VMM=1 implies it): event records, event waits and
// graph launches, logged in order with the launches as {"ev": "record" | "wait" |
// "graph_launch", ...} lines carrying a launch's stream fields (stream_id, dev, capturing,
// capture_id) and the call's flags, a capturing stream's rewritten dependencies
// (cuStreamUpdateCaptureDependencies) as {"ev": "capture_deps", ...} with the same fields, and
// event destructions, IPC event opens and IPC event exports as {"ev": "destroy" | "ipc_open" |
// "ipc_get", "event": ...} (after a destruction the handle no longer names the event recorded
// under it; an event shared with another process can be recorded there). With each launch's
// stream they are the order CUDA enforces between the launches (python/ptxv/_order.py,
// docs/capture.md "Launch order"). A call is logged after it returned, and only when it
// succeeded (a failed wait orders nothing). Each flavour of an entry point (legacy, _ptsz) has
// its own hook and real pointer, installed through cuGetProcAddress (by its flags), dlsym (by
// name) and the exported names (direct calls).
static int order_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("LAUNCHCAP_ORDER");
        on = v ? atoi(v) != 0 : getenv("LAUNCHCAP_VMM") != 0;
    }
    return on;
}
typedef CUresult (*evrec_t)(CUevent, CUstream);
typedef CUresult (*evrecf_t)(CUevent, CUstream, unsigned);
typedef CUresult (*swait_t)(CUstream, CUevent, unsigned);
typedef CUresult (*glaunch_t)(CUgraphExec, CUstream);
typedef CUresult (*evdestroy_t)(CUevent);
typedef CUresult (*evipcopen_t)(CUevent *, CUipcEventHandle);
typedef CUresult (*evipcget_t)(CUipcEventHandle *, CUevent);
typedef CUresult (*ucd1_t)(CUstream, CUgraphNode *, size_t, unsigned);
typedef CUresult (*ucd2_t)(CUstream, CUgraphNode *, const CUgraphEdgeData *, size_t, unsigned);
static evrec_t r_evrec[2];
static evrecf_t r_evrecf[2];
static swait_t r_swait[2];
static glaunch_t r_glaunch[2];
// No per-thread flavour. [0]: cuEventDestroy under its first name (cuGetProcAddress hands the
// version the caller asked for under it), [1]: cuEventDestroy_v2.
static evdestroy_t r_evdestroy[2];
static evipcopen_t r_evipcopen;
static evipcget_t r_evipcget;
// cuStreamUpdateCaptureDependencies, first version and _v2, each in both flavours.
static ucd1_t r_ucd1[2];
static ucd2_t r_ucd2[2];

// "t" = "t0": host time just before the call; "t1": after it returned.
static void ev_log(const char *kind, const char *what, unsigned long long h, unsigned flags,
                   CUstream s, int ptsz, unsigned long long t0, unsigned long long t1) {
    struct sinfo si = stream_info(s, ptsz);
    FILE *fp = out_begin();
    if (!fp) return;
    fprintf(fp, "{\"ev\": \"%s\", \"%s\": %llu, \"flags\": %u, \"stream\": %llu, \"dev\": %d, ",
            kind, what, h, flags, (unsigned long long)s, si.dev);
    put_stream_info(fp, si, ptsz);
    fprintf(fp, "\"t\": %llu, \"t0\": %llu, \"t1\": %llu, \"tid\": %ld}\n", t0, t0, t1, tid());
    out_end(fp);
}
// An event handle's end (no stream: the call is not stream work).
static void ev_log_handle(const char *kind, unsigned long long h, unsigned long long t0,
                          unsigned long long t1) {
    int dev = cur_device();
    FILE *fp = out_begin();
    if (!fp) return;
    fprintf(fp, "{\"ev\": \"%s\", \"event\": %llu, \"dev\": %d, \"t\": %llu, \"t0\": %llu, "
                "\"t1\": %llu, \"tid\": %ld}\n",
            kind, h, dev, t0, t0, t1, tid());
    out_end(fp);
}

#define ORDER_REAL(arr, ptsz, name)                                                   \
    if (!arr[ptsz]) arr[ptsz] = (__typeof__(arr[0]))next_sym(ptsz ? name "_ptsz" : name)
static CUresult evrec_hook(int ptsz, CUevent e, CUstream s) {
    ORDER_REAL(r_evrec, ptsz, "cuEventRecord");
    unsigned long long t0 = now_ns();
    CUresult r = r_evrec[ptsz](e, s);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on())
        ev_log("record", "event", (unsigned long long)e, 0, s, ptsz, t0, t1);
    return r;
}
static CUresult evrecf_hook(int ptsz, CUevent e, CUstream s, unsigned fl) {
    ORDER_REAL(r_evrecf, ptsz, "cuEventRecordWithFlags");
    unsigned long long t0 = now_ns();
    CUresult r = r_evrecf[ptsz](e, s, fl);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on())
        ev_log("record", "event", (unsigned long long)e, fl, s, ptsz, t0, t1);
    return r;
}
static CUresult swait_hook(int ptsz, CUstream s, CUevent e, unsigned fl) {
    ORDER_REAL(r_swait, ptsz, "cuStreamWaitEvent");
    unsigned long long t0 = now_ns();
    CUresult r = r_swait[ptsz](s, e, fl);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on())
        ev_log("wait", "event", (unsigned long long)e, fl, s, ptsz, t0, t1);
    return r;
}
static CUresult glaunch_hook(int ptsz, CUgraphExec g, CUstream s) {
    ORDER_REAL(r_glaunch, ptsz, "cuGraphLaunch");
    unsigned long long t0 = now_ns();
    CUresult r = r_glaunch[ptsz](g, s);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on())
        ev_log("graph_launch", "graph", (unsigned long long)g, 0, s, ptsz, t0, t1);
    return r;
}
static CUresult evdestroy_hook(int v2, CUevent e) {
    if (!r_evdestroy[v2])
        r_evdestroy[v2] = (evdestroy_t)next_sym(v2 ? "cuEventDestroy_v2" : "cuEventDestroy");
    unsigned long long t0 = now_ns();
    CUresult r = r_evdestroy[v2](e);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on()) ev_log_handle("destroy", (unsigned long long)e, t0, t1);
    return r;
}
CUresult cuEventDestroy(CUevent e) { return evdestroy_hook(0, e); }
CUresult cuEventDestroy_v2(CUevent e) { return evdestroy_hook(1, e); }
CUresult cuIpcOpenEventHandle(CUevent *pe, CUipcEventHandle h) {
    if (!r_evipcopen) r_evipcopen = (evipcopen_t)next_sym("cuIpcOpenEventHandle");
    unsigned long long t0 = now_ns();
    CUresult r = r_evipcopen(pe, h);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && pe && order_on())
        ev_log_handle("ipc_open", (unsigned long long)*pe, t0, t1);
    return r;
}
static CUresult ucd1_hook(int ptsz, CUstream s, CUgraphNode *d, size_t n, unsigned fl) {
    ORDER_REAL(r_ucd1, ptsz, "cuStreamUpdateCaptureDependencies");
    unsigned long long t0 = now_ns();
    CUresult r = r_ucd1[ptsz](s, d, n, fl);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on()) ev_log("capture_deps", "nodes", n, fl, s, ptsz, t0, t1);
    return r;
}
static CUresult ucd2_hook(int ptsz, CUstream s, CUgraphNode *d, const CUgraphEdgeData *e, size_t n,
                          unsigned fl) {
    ORDER_REAL(r_ucd2, ptsz, "cuStreamUpdateCaptureDependencies_v2");
    unsigned long long t0 = now_ns();
    CUresult r = r_ucd2[ptsz](s, d, e, n, fl);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on()) ev_log("capture_deps", "nodes", n, fl, s, ptsz, t0, t1);
    return r;
}
CUresult cuStreamUpdateCaptureDependencies(CUstream s, CUgraphNode *d, size_t n, unsigned fl) {
    return ucd1_hook(0, s, d, n, fl);
}
CUresult cuStreamUpdateCaptureDependencies_ptsz(CUstream s, CUgraphNode *d, size_t n, unsigned fl) {
    return ucd1_hook(1, s, d, n, fl);
}
CUresult cuStreamUpdateCaptureDependencies_v2(CUstream s, CUgraphNode *d, const CUgraphEdgeData *e,
                                              size_t n, unsigned fl) {
    return ucd2_hook(0, s, d, e, n, fl);
}
CUresult cuStreamUpdateCaptureDependencies_v2_ptsz(CUstream s, CUgraphNode *d,
                                                   const CUgraphEdgeData *e, size_t n, unsigned fl) {
    return ucd2_hook(1, s, d, e, n, fl);
}
CUresult cuIpcGetEventHandle(CUipcEventHandle *ph, CUevent e) {
    if (!r_evipcget) r_evipcget = (evipcget_t)next_sym("cuIpcGetEventHandle");
    unsigned long long t0 = now_ns();
    CUresult r = r_evipcget(ph, e);
    unsigned long long t1 = now_ns();
    if (r == CUDA_SUCCESS && order_on()) ev_log_handle("ipc_get", (unsigned long long)e, t0, t1);
    return r;
}
CUresult cuEventRecord(CUevent e, CUstream s) { return evrec_hook(0, e, s); }
CUresult cuEventRecord_ptsz(CUevent e, CUstream s) { return evrec_hook(1, e, s); }
CUresult cuEventRecordWithFlags(CUevent e, CUstream s, unsigned fl) { return evrecf_hook(0, e, s, fl); }
CUresult cuEventRecordWithFlags_ptsz(CUevent e, CUstream s, unsigned fl) {
    return evrecf_hook(1, e, s, fl);
}
CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned fl) { return swait_hook(0, s, e, fl); }
CUresult cuStreamWaitEvent_ptsz(CUstream s, CUevent e, unsigned fl) { return swait_hook(1, s, e, fl); }
CUresult cuGraphLaunch(CUgraphExec g, CUstream s) { return glaunch_hook(0, g, s); }
CUresult cuGraphLaunch_ptsz(CUgraphExec g, CUstream s) { return glaunch_hook(1, g, s); }

// Swap an entry point (`sym` without its _ptsz suffix; `ptsz`: which flavour `p` is) for its
// hook, remembering the real one: the launches always, stream order calls when recorded.
static void *wrap_launch_order(const char *sym, void *p, int ptsz) {
#define ORDER_HOOK(name, arr, hook, hook_ptsz)                       \
    if (!strcmp(sym, name)) {                                        \
        void *h = ptsz ? (void *)hook_ptsz : (void *)hook;           \
        if (p == h || p == (void *)hook || p == (void *)hook_ptsz) return p; \
        arr[ptsz] = (__typeof__(arr[0]))p;                           \
        return h;                                                    \
    }
    ORDER_HOOK("cuLaunchKernel", real_launch, cuLaunchKernel, cuLaunchKernel_ptsz)
    ORDER_HOOK("cuLaunchKernelEx", real_launchex, cuLaunchKernelEx, cuLaunchKernelEx_ptsz)
    if (!order_on()) return p;
    ORDER_HOOK("cuEventRecord", r_evrec, cuEventRecord, cuEventRecord_ptsz)
    ORDER_HOOK("cuEventRecordWithFlags", r_evrecf, cuEventRecordWithFlags, cuEventRecordWithFlags_ptsz)
    ORDER_HOOK("cuStreamWaitEvent", r_swait, cuStreamWaitEvent, cuStreamWaitEvent_ptsz)
    ORDER_HOOK("cuGraphLaunch", r_glaunch, cuGraphLaunch, cuGraphLaunch_ptsz)
#undef ORDER_HOOK
    if (!strcmp(sym, "cuEventDestroy") || !strcmp(sym, "cuEventDestroy_v2")) {
        int v2 = sym[14] != 0;
        if (p == (void *)cuEventDestroy || p == (void *)cuEventDestroy_v2) return p;
        r_evdestroy[v2] = (evdestroy_t)p;
        return v2 ? (void *)cuEventDestroy_v2 : (void *)cuEventDestroy;
    }
    if (!strcmp(sym, "cuIpcOpenEventHandle")) {
        if (p == (void *)cuIpcOpenEventHandle) return p;
        r_evipcopen = (evipcopen_t)p;
        return (void *)cuIpcOpenEventHandle;
    }
    if (!strcmp(sym, "cuIpcGetEventHandle")) {
        if (p == (void *)cuIpcGetEventHandle) return p;
        r_evipcget = (evipcget_t)p;
        return (void *)cuIpcGetEventHandle;
    }
    if (!strcmp(sym, "cuStreamUpdateCaptureDependencies") ||
        !strcmp(sym, "cuStreamUpdateCaptureDependencies_v2")) {
        // cuGetProcAddress hands the version the caller asked for under the first name: tell
        // the two apart by the driver's own entry points.
        void *v2h[2] = {(void *)cuStreamUpdateCaptureDependencies_v2,
                        (void *)cuStreamUpdateCaptureDependencies_v2_ptsz};
        void *v1h[2] = {(void *)cuStreamUpdateCaptureDependencies,
                        (void *)cuStreamUpdateCaptureDependencies_ptsz};
        if (p == v2h[0] || p == v2h[1] || p == v1h[0] || p == v1h[1]) return p;
        int v2 = sym[33] != 0;
        if (!v2) {
            void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
            init_real_dlsym();
            v2 = cuda && (p == real_dlsym(cuda, "cuStreamUpdateCaptureDependencies_v2") ||
                          p == real_dlsym(cuda, "cuStreamUpdateCaptureDependencies_v2_ptsz"));
        }
        if (v2) r_ucd2[ptsz] = (ucd2_t)p;
        else r_ucd1[ptsz] = (ucd1_t)p;
        return v2 ? v2h[ptsz] : v1h[ptsz];
    }
    return p;
}

// Runtimes (cudart, CUDA Python) find driver entry points through cuGetProcAddress, which
// they in turn look up with dlsym on libcuda's handle: intercept both.
typedef CUresult (*gpa_t)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
static gpa_t real_gpa;

static CUresult hook_gpa(const char *sym, void **pfn, int ver, cuuint64_t flags,
                         CUdriverProcAddressQueryResult *st) {
    CUresult r = real_gpa(sym, pfn, ver, flags, st);
    if (getenv("LAUNCHCAP_DEBUG")) fprintf(stderr, "launchcap: gpa %s\n", sym);
    if (r == CUDA_SUCCESS && pfn && *pfn) {
        if (!strcmp(sym, "cuGetProcAddress")) {
            // cudart looks up every driver entry point it uses through the cuGetProcAddress it
            // got from cuGetProcAddress (torch's DriverAPI: cudaGetDriverEntryPointByVersion;
            // cuBLAS's statically linked cudart: its gemmSN / gemv kernels for small shapes
            // were not recorded at all without this).
            *pfn = (void *)hook_gpa;
        } else {
            // The flags pick the flavour (cuda.h sets the per-thread one explicitly for code
            // built with a per-thread default stream; 0 is the legacy one).
            *pfn = wrap_launch_order(sym, *pfn,
                                     (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0);
            if (getenv("LAUNCHCAP_IMAGES")) *pfn = wrap_loader(sym, *pfn);
            if (getenv("LAUNCHCAP_VMM")) *pfn = wrap_vmm(sym, *pfn);
        }
    }
    return r;
}

void *dlsym(void *h, const char *sym) {
    init_real_dlsym();
    void *p = real_dlsym(h, sym);
    if (p && sym && !strncmp(sym, "cuGetProcAddress", 16)) {
        real_gpa = (gpa_t)p;
        return (void *)hook_gpa;
    }
    if (p && sym && !strncmp(sym, "cu", 2)) {
        // By name: "<entry point>" or "<entry point>_ptsz".
        char base[128];
        size_t n = strlen(sym);
        int ptsz = n > 5 && n < sizeof base && !strcmp(sym + n - 5, "_ptsz");
        snprintf(base, sizeof base, "%.*s", (int)(ptsz ? n - 5 : n), sym);
        p = wrap_launch_order(base, p, ptsz);
    }
    if (p && sym && getenv("LAUNCHCAP_IMAGES") && !strncmp(sym, "cu", 2)) p = wrap_loader(sym, p);
    if (p && sym && getenv("LAUNCHCAP_VMM") && !strncmp(sym, "cu", 2)) p = wrap_vmm(sym, p);
    return p;
}

// CuTeDSL's JIT host code launches through libcute_dsl_runtime's `_cudaLaunchKernelEx`
// (cudaLaunchKernelExC on a cudaKernel_t from cudaLibraryGetKernel).
cudaError_t _cudaLaunchKernelEx(const cudaLaunchConfig_t *c, const void *k, void **args) {
    static cudaError_t (*real)(const cudaLaunchConfig_t *, const void *, void **);
    if (!real) real = dlsym(RTLD_NEXT, "_cudaLaunchKernelEx");
    unsigned cx = 1, cy = 1, cz = 1;
    for (unsigned i = 0; i < c->numAttrs; i++)
        if (c->attrs[i].id == cudaLaunchAttributeClusterDimension) {
            cx = c->attrs[i].val.clusterDim.x;
            cy = c->attrs[i].val.clusterDim.y;
            cz = c->attrs[i].val.clusterDim.z;
        } else if (c->attrs[i].id == cudaLaunchAttributeProgrammaticStreamSerialization) {
            cur_pdl = c->attrs[i].val.programmaticStreamSerializationAllowed != 0;
        } else if (c->attrs[i].id == cudaLaunchAttributePreferredClusterDimension) {
            cur_pref[0] = c->attrs[i].val.preferredClusterDim.x;
            cur_pref[1] = c->attrs[i].val.preferredClusterDim.y;
            cur_pref[2] = c->attrs[i].val.preferredClusterDim.z;
        } else {
            order_attr_rt(&c->attrs[i]);
        }
    cur_stream = (unsigned long long)c->stream;
    snap_allowed = 1;
    dump_k((CUkernel)k, 0, c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x,
           c->blockDim.y, c->blockDim.z, (unsigned)c->dynamicSmemBytes, cx, cy, cz, args);
    snap_allowed = 0;
    cur_pdl = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    reset_order_attrs();
    snap_pre();
    unsigned long long t0 = now_ns();
    in_rt++;
    cudaError_t r = real(c, k, args);
    in_rt--;
    unsigned long long t1 = now_ns();
    snap_post();
    launch_done(r == cudaSuccess, t0, t1);
    return r;
}

// PyTorch's own kernels launch through the CUDA runtime it links dynamically (libcudart):
// `<<<>>>` calls __cudaLaunchKernel, CUTLASS's launcher cudaLaunchKernelExC. Their kernel
// handle is the host stub; cudaGetFuncBySymbol gives the driver function. Everything is
// looked up in the libcudart already loaded (not linked here: a second copy of the runtime
// has no kernels registered).
static void *cudart(const char *sym) {
    void *h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
    init_real_dlsym();
    return h ? real_dlsym(h, sym) : 0;
}

// Called after the launch (the kernel is loaded by then), or before it when snapshots are asked
// for: the driver function, or else the library-managed kernel (`*kern`). A failed query would
// leave its error as the runtime's last error, which PyTorch checks after every launch: clear
// it.
static CUfunction stub_func(const void *k, CUkernel *kern) {
    typedef int (*gfbs_t)(void **, const void *);
    typedef int (*gle_t)(void);
    gfbs_t g = (gfbs_t)cudart("cudaGetFuncBySymbol");
    gfbs_t gk = (gfbs_t)cudart("cudaGetKernel");
    gle_t gle = (gle_t)cudart("cudaGetLastError");
    void *f = 0;
    *kern = 0;
    if (g && g(&f, k) == 0) return (CUfunction)f;
    if (gle) gle();
    if (gk && gk(&f, k) == 0) {
        *kern = (CUkernel)f;
        return 0;
    }
    if (gle) gle();
    // CUDA 13's __cudaLaunchKernel may pass the cudaKernel_t itself.
    {
        typedef CUresult (*kname_t)(const char **, CUkernel);
        void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        kname_t kn = cuda ? (kname_t)real_dlsym(cuda, "cuKernelGetName") : 0;
        const char *nm = 0;
        if (kn && kn(&nm, (CUkernel)k) == CUDA_SUCCESS) {
            *kern = (CUkernel)k;
            return 0;
        }
    }
    if (getenv("LAUNCHCAP_DEBUG")) fprintf(stderr, "launchcap: no function for stub %p\n", k);
    return 0;
}

cudaError_t __cudaLaunchKernel(const void *k, dim3 g, dim3 b, void **args, size_t smem, cudaStream_t s) {
    typedef cudaError_t (*real_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static real_t real;
    if (!real) real = (real_t)cudart("__cudaLaunchKernel");
    if (snap_wanted()) {
        // Snapshots asked for: the function is looked up first (the kernel's module is loaded
        // once the program ran it, or with CUDA_MODULE_LOADING=EAGER), so that the launch is
        // recorded, and its allocations copied, before it runs.
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        if (f || kk) {
            cur_stream = (unsigned long long)s;
            snap_allowed = 1;
            dump_k(kk, f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)smem, 1, 1, 1, args);
            snap_allowed = 0;
            snap_pre();
            unsigned long long t0 = now_ns();
            in_rt++;
            cudaError_t r = real(k, g, b, args, smem, s);
            in_rt--;
            unsigned long long t1 = now_ns();
            snap_post();
            launch_done(r == cudaSuccess, t0, t1);
            return r;
        }
    }
    unsigned long long t0 = now_ns();
    in_rt++;
    cudaError_t r = real(k, g, b, args, smem, s);
    in_rt--;
    unsigned long long t1 = now_ns();
    if (r == cudaSuccess) {
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        cur_stream = (unsigned long long)s;
        if (f || kk) {
            dump_k(kk, f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)smem, 1, 1, 1, args);
            launch_done(1, t0, t1);
        }
    }
    return r;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c, const void *k, void **args) {
    typedef cudaError_t (*real_t)(const cudaLaunchConfig_t *, const void *, void **);
    static real_t real;
    if (!real) real = (real_t)cudart("cudaLaunchKernelExC");
    unsigned cx = 1, cy = 1, cz = 1;
    for (unsigned i = 0; i < c->numAttrs; i++)
        if (c->attrs[i].id == cudaLaunchAttributeClusterDimension) {
            cx = c->attrs[i].val.clusterDim.x;
            cy = c->attrs[i].val.clusterDim.y;
            cz = c->attrs[i].val.clusterDim.z;
        } else if (c->attrs[i].id == cudaLaunchAttributeProgrammaticStreamSerialization) {
            cur_pdl = c->attrs[i].val.programmaticStreamSerializationAllowed != 0;
        } else if (c->attrs[i].id == cudaLaunchAttributePreferredClusterDimension) {
            cur_pref[0] = c->attrs[i].val.preferredClusterDim.x;
            cur_pref[1] = c->attrs[i].val.preferredClusterDim.y;
            cur_pref[2] = c->attrs[i].val.preferredClusterDim.z;
        } else {
            order_attr_rt(&c->attrs[i]);
        }
    if (snap_wanted()) {
        // As __cudaLaunchKernel: recorded and snapshotted before the launch.
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        if (f || kk) {
            cur_stream = (unsigned long long)c->stream;
            snap_allowed = 1;
            dump_k(kk, f, c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x, c->blockDim.y,
                   c->blockDim.z, (unsigned)c->dynamicSmemBytes, cx, cy, cz, args);
            snap_allowed = 0;
            cur_pdl = 0;
            memset(cur_pref, 0, sizeof cur_pref);
            reset_order_attrs();
            snap_pre();
            unsigned long long t0 = now_ns();
            in_rt++;
            cudaError_t r = real(c, k, args);
            in_rt--;
            unsigned long long t1 = now_ns();
            snap_post();
            launch_done(r == cudaSuccess, t0, t1);
            return r;
        }
    }
    unsigned long long t0 = now_ns();
    in_rt++;
    cudaError_t r = real(c, k, args);
    in_rt--;
    unsigned long long t1 = now_ns();
    if (r == cudaSuccess) {
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        cur_stream = (unsigned long long)c->stream;
        if (f || kk) {
            dump_k(kk, f, c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x, c->blockDim.y,
                   c->blockDim.z, (unsigned)c->dynamicSmemBytes, cx, cy, cz, args);
            launch_done(1, t0, t1);
        }
    }
    cur_pdl = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    reset_order_attrs();
    return r;
}

// Memory operations on the stream (cudaMemcpy* / cudaMemset*: an op's `clone()` or zeroing is
// not a kernel), recorded in order with the launches when LAUNCHCAP_MEMOPS=1, so a capture
// can account for them (tools/capture.py).
static void log_memop(const char *op, const void *dst, const void *src, int value, size_t n,
                      int kind) {
    if (!getenv("LAUNCHCAP_MEMOPS")) return;
    FILE *fp = out_begin();
    if (!fp) return;
    fprintf(fp, "{\"op\": \"%s\", \"dst\": %llu, \"src\": %llu, \"value\": %d, "
                "\"bytes\": %zu, \"kind\": %d, \"t\": %llu",
            op, (unsigned long long)dst, (unsigned long long)src, value, n, kind, now_ns());
    // Host-to-device copies carry their bytes (a host-computed index table uploaded between
    // kernels: tools/capture_dist.py folds it into the destination's contents).
    if (kind == 1 && src && n <= (1u << 20)) {
        fprintf(fp, ", \"data\": \"");
        for (size_t i = 0; i < n; i++) fprintf(fp, "%02x", ((const unsigned char *)src)[i]);
        fprintf(fp, "\"");
    }
    fprintf(fp, "}\n");
    out_end(fp);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, enum cudaMemcpyKind k,
                            cudaStream_t s) {
    typedef cudaError_t (*real_t)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static real_t real;
    if (!real) real = (real_t)cudart("cudaMemcpyAsync");
    log_memop("memcpy", dst, src, 0, n, k);
    return real(dst, src, n, k, s);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, enum cudaMemcpyKind k) {
    typedef cudaError_t (*real_t)(void *, const void *, size_t, enum cudaMemcpyKind);
    static real_t real;
    if (!real) real = (real_t)cudart("cudaMemcpy");
    log_memop("memcpy", dst, src, 0, n, k);
    return real(dst, src, n, k);
}

cudaError_t cudaMemsetAsync(void *dst, int value, size_t n, cudaStream_t s) {
    typedef cudaError_t (*real_t)(void *, int, size_t, cudaStream_t);
    static real_t real;
    if (!real) real = (real_t)cudart("cudaMemsetAsync");
    log_memop("memset", dst, 0, value, n, 0);
    return real(dst, value, n, s);
}

cudaError_t cudaMemset(void *dst, int value, size_t n) {
    typedef cudaError_t (*real_t)(void *, int, size_t);
    static real_t real;
    if (!real) real = (real_t)cudart("cudaMemset");
    log_memop("memset", dst, 0, value, n, 0);
    return real(dst, value, n);
}
