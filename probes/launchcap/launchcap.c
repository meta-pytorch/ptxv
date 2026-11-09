/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// LD_PRELOAD launch capture: for every cuLaunchKernel / cuLaunchKernelEx, append the grid,
// block, dynamic smem, cluster dims and every parameter's bytes (sizes from
// cuFuncGetParamInfo) to $LAUNCHCAP_OUT as one JSON object per line. Used to turn a JIT
// framework's launch (CuTeDSL, binary libraries) into a ptxv case.
//   gcc -shared -fPIC -O2 -o launchcap.so launchcap.c -ldl -I/usr/local/cuda-13.0/include
#define _GNU_SOURCE
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef CUresult (*getparam_t)(CUfunction, size_t, size_t *, size_t *);
typedef CUresult (*kgetparam_t)(CUkernel, size_t, size_t *, size_t *);
typedef CUresult (*getname_t)(const char **, CUfunction);

// The `extra` argument of the launch being recorded (libraries such as cuBLAS pass their
// arguments as one packed buffer: CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE).
static void **cur_extra;
// Whether the launch being recorded carries the programmatic stream serialization attribute
// (PDL: it may start before the previous kernel on the stream completes).
static int cur_pdl;
// The preferred cluster dimension of the launch being recorded (0s without the attribute).
// With it, the hardware launches each group of CTAs either as one cluster of this shape or as
// clusters of the regular (fallback) shape (probes/cga_size).
static unsigned cur_pref[3];
// The stream of the launch being recorded (multi-process captures order launches per stream).
static unsigned long long cur_stream;
// Inside a runtime launch (__cudaLaunchKernel / cudaLaunchKernelExC), which records the launch
// itself: the driver launch it makes underneath is not recorded again.
static __thread int in_rt;

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
// wrote. Only for launches recorded before they are issued (driver launches,
// _cudaLaunchKernelEx).
static int snap_on;
static __thread int snap_allowed;  // set by the hooks that record before launching
static unsigned snap_seq;
static unsigned char snap_buf[1 << 16];
static size_t snap_len;
static int snap_match(const char *nm) {
    static int state;  // 0 untried, 1 compiled, -1 off
    static regex_t re;
    if (!state) {
        const char *p = getenv("LAUNCHCAP_SNAP");
        state = p && !regcomp(&re, p, REG_EXTENDED | REG_NOSUB) ? 1 : -1;
    }
    return state == 1 && !regexec(&re, nm, 0, 0, 0);
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
            snprintf(path, sizeof path, "%s/s%u_%s_%llx.bin", dir ? dir : ".", snap_seq, tag,
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
    if (off + n <= sizeof snap_buf) {
        memcpy(snap_buf + off, p, n);
        if (off + n > snap_len) snap_len = off + n;
    }
}

static FILE *open_out(void) {
    char buf[4096];
    return fopen(rank_path("LAUNCHCAP_OUT", "launchcap.jsonl", buf, sizeof buf), "a");
}

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

static void dump_k(CUkernel k, CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                 unsigned by, unsigned bz, unsigned smem, unsigned cx, unsigned cy,
                 unsigned cz, void **params) {
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    getparam_t info = (getparam_t)dlsym(cuda, "cuFuncGetParamInfo");
    kgetparam_t kinfo = (kgetparam_t)dlsym(cuda, "cuKernelGetParamInfo");
    getname_t name = (getname_t)dlsym(cuda, "cuFuncGetName");
    FILE *fp = open_out();
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
    snap_on = snap_allowed && snap_match(nm);
    snap_len = 0;
    if (snap_on) fprintf(fp, "{\"snap\": %u, ", snap_seq);
    fprintf(fp, "%s\"name\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
                "\"smem\": %u, \"cluster\": [%u, %u, %u], \"pdl\": %d, \"stream\": %llu, "
                "\"dev\": %d, \"t\": %llu, \"mod\": %llu, \"lib\": %llu, ",
            snap_on ? "" : "{", nm, gx, gy, gz, bx, by, bz, smem, cx, cy, cz, cur_pdl, cur_stream, cur_device(),
            now_ns(), hmod, hlib);
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
    fprintf(fp, "]}\n");
    fclose(fp);
}

#define dump(f, ...) dump_k(0, f, __VA_ARGS__)

static void snap_pre(void) {
    if (snap_on) snap_dump("pre");
}
static void snap_post(void) {
    if (!snap_on) return;
    snap_dump("post");
    snap_on = 0;
    snap_seq++;
}

typedef CUresult (*launch_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, unsigned, CUstream, void **, void **);
typedef CUresult (*launchex_t)(const CUlaunchConfig *, CUfunction, void **, void **);
static launch_t real_launch;
static launchex_t real_launchex;

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                        unsigned by, unsigned bz, unsigned smem, CUstream s, void **params,
                        void **extra) {
    if (!real_launch) real_launch = (launch_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
    if (in_rt) return real_launch(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
    cur_extra = extra;
    cur_stream = (unsigned long long)s;
    snap_allowed = 1;
    dump(f, gx, gy, gz, bx, by, bz, smem, 1, 1, 1, params);
    snap_allowed = 0;
    cur_extra = 0;
    snap_pre();
    CUresult r = real_launch(f, gx, gy, gz, bx, by, bz, smem, s, params, extra);
    snap_post();
    return r;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **params, void **extra) {
    if (!real_launchex) real_launchex = (launchex_t)dlsym(RTLD_NEXT, "cuLaunchKernelEx");
    if (in_rt) return real_launchex(c, f, params, extra);
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
        }
    cur_extra = extra;
    cur_stream = (unsigned long long)c->hStream;
    snap_allowed = 1;
    dump(f, c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY, c->blockDimZ,
         c->sharedMemBytes, cx, cy, cz, params);
    snap_allowed = 0;
    cur_extra = 0;
    cur_pdl = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    snap_pre();
    CUresult r = real_launchex(c, f, params, extra);
    snap_post();
    return r;
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

// Virtual memory management, IPC, multicast and stream-ordering calls (LAUNCHCAP_VMM=1),
// logged in order with the launches as {"vmm": ...} / {"ev": ...} lines. A multi-process
// capture (tools/capture_dist.py) uses them to resolve a pointer into a peer's memory: torch's
// symmetric memory exports each rank's cuMemCreate allocation and maps it into every peer
// (cuMemImportFromShareableHandle + cuMemMap), and binds it to a multicast object (NVLS).
static void vlog(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#include <stdarg.h>
static void vlog(const char *fmt, ...) {
    FILE *fp = open_out();
    if (!fp) return;
    va_list a;
    va_start(a, fmt);
    vfprintf(fp, fmt, a);
    va_end(a);
    fprintf(fp, "\n");
    fclose(fp);
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
typedef CUresult (*evrec_t)(CUevent, CUstream);
typedef CUresult (*evrecf_t)(CUevent, CUstream, unsigned);
typedef CUresult (*swait_t)(CUstream, CUevent, unsigned);
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
static evrec_t r_evrec;
static evrecf_t r_evrecf;
static swait_t r_swait;
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
static CUresult h_evrec(CUevent e, CUstream s) {
    vlog("{\"ev\": \"record\", \"event\": %llu, \"stream\": %llu, \"t\": %llu}",
         (unsigned long long)e, (unsigned long long)s, now_ns());
    return r_evrec(e, s);
}
static CUresult h_evrecf(CUevent e, CUstream s, unsigned fl) {
    vlog("{\"ev\": \"record\", \"event\": %llu, \"stream\": %llu, \"t\": %llu}",
         (unsigned long long)e, (unsigned long long)s, now_ns());
    return r_evrecf(e, s, fl);
}
static CUresult h_swait(CUstream s, CUevent e, unsigned fl) {
    vlog("{\"ev\": \"wait\", \"event\": %llu, \"stream\": %llu, \"t\": %llu}",
         (unsigned long long)e, (unsigned long long)s, now_ns());
    return r_swait(s, e, fl);
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
    VMM_HOOK("cuEventRecord", r_evrec, h_evrec)
    VMM_HOOK("cuEventRecordWithFlags", r_evrecf, h_evrecf)
    VMM_HOOK("cuStreamWaitEvent", r_swait, h_swait)
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
        } else if (!strcmp(sym, "cuLaunchKernel")) {
            real_launch = (launch_t)*pfn;
            *pfn = (void *)cuLaunchKernel;
        } else if (!strcmp(sym, "cuLaunchKernelEx")) {
            real_launchex = (launchex_t)*pfn;
            *pfn = (void *)cuLaunchKernelEx;
        } else {
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
    if (p && sym && !strcmp(sym, "cuLaunchKernelEx") && p != (void *)cuLaunchKernelEx) {
        real_launchex = (launchex_t)p;
        return (void *)cuLaunchKernelEx;
    }
    if (p && sym && !strcmp(sym, "cuLaunchKernel") && p != (void *)cuLaunchKernel) {
        real_launch = (launch_t)p;
        return (void *)cuLaunchKernel;
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
        }
    cur_stream = (unsigned long long)c->stream;
    snap_allowed = 1;
    dump_k((CUkernel)k, 0, c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x,
           c->blockDim.y, c->blockDim.z, (unsigned)c->dynamicSmemBytes, cx, cy, cz, args);
    snap_allowed = 0;
    cur_pdl = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    snap_pre();
    in_rt++;
    cudaError_t r = real(c, k, args);
    in_rt--;
    snap_post();
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

// Called after the launch (the kernel is loaded by then): the driver function, or else the
// library-managed kernel (`*kern`). A failed query would leave its error as the runtime's
// last error, which PyTorch checks after every launch: clear it.
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
    in_rt++;
    cudaError_t r = real(k, g, b, args, smem, s);
    in_rt--;
    if (r == cudaSuccess) {
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        cur_stream = (unsigned long long)s;
        if (f || kk) dump_k(kk, f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)smem, 1, 1, 1, args);
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
        }
    in_rt++;
    cudaError_t r = real(c, k, args);
    in_rt--;
    if (r == cudaSuccess) {
        CUkernel kk;
        CUfunction f = stub_func(k, &kk);
        cur_stream = (unsigned long long)c->stream;
        if (f || kk)
            dump_k(kk, f, c->gridDim.x, c->gridDim.y, c->gridDim.z, c->blockDim.x, c->blockDim.y,
                   c->blockDim.z, (unsigned)c->dynamicSmemBytes, cx, cy, cz, args);
    }
    cur_pdl = 0;
    memset(cur_pref, 0, sizeof cur_pref);
    return r;
}

// Memory operations on the stream (cudaMemcpy* / cudaMemset*: an op's `clone()` or zeroing is
// not a kernel), recorded in order with the launches when LAUNCHCAP_MEMOPS=1, so a capture
// can account for them (tools/capture.py).
static void log_memop(const char *op, const void *dst, const void *src, int value, size_t n,
                      int kind) {
    if (!getenv("LAUNCHCAP_MEMOPS")) return;
    FILE *fp = open_out();
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
    fclose(fp);
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
