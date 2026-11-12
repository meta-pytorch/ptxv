/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* LD_PRELOAD shim for outdir_race.rs: two faults in the way a run locks its own run directory
 * (main.rs `lock_run_dir`: create `.lock.new`, flock it, rename it to `.lock`) and one in the way
 * it reaps a dead run's (`remove_run_dir`). The lock faults act only on this process's own run
 * directory (a path holding "-<pid>-", the pid /proc/self names) and only on files whose last
 * component starts with ".lock", the reaping fault only on directories of the name it is given;
 * everything else goes through unchanged.
 *
 *   SHIM_FLOCK_DELAY_MS=N   flock(LOCK_EX) on the lock file is delayed N ms
 *                           ("shim: flock(<path>) delayed N ms"): a wide window between the
 *                           file's creation and its lock (a_run_taking_its_lock_keeps_its_directory).
 *   SHIM_RENAME_LOST_MS=N   the rename of `.lock.new` is done and reported failed (ENOENT, as
 *                           when an NFS reply is lost after the server renamed), and the unlink
 *                           of `.lock.new` that follows is delayed N ms ("shim: rename(<path>)
 *                           reported lost; unlink delayed N ms")
 *                           (a_run_whose_lock_rename_is_reported_failed_keeps_its_directory).
 *   SHIM_VANISH_DIR=NAME    opendir of a directory named NAME removes it first (rmdir: it is
 *                           empty), as another run's reaper does between this run's listing of
 *                           its parent and its reading of it ("shim: opendir(<path>) removed
 *                           first") (a_subdirectory_removed_under_a_reaper_counts_as_gone).
 *
 *   cc -shared -fPIC -o shim_flock.so shim_flock.c -ldl */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

/* Whether `path` is a lock file of this process's own run directory. */
static int own(const char *path) {
    char self[64], me[80];
    ssize_t m = readlink("/proc/self", self, sizeof self - 1);
    if (m <= 0 || !path) {
        return 0;
    }
    self[m] = 0;
    snprintf(me, sizeof me, "-%s-", self);
    const char *last = strrchr(path, '/');
    return strstr(path, me) && last && strncmp(last, "/.lock", 6) == 0;
}

static int is_new(const char *path) {
    size_t n = path ? strlen(path) : 0;
    return n >= 10 && strcmp(path + n - 10, "/.lock.new") == 0;
}

static void nap(const char *ms_text) {
    int ms = atoi(ms_text);
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

int flock(int fd, int op) {
    static int (*real)(int, int);
    if (!real) {
        real = (int (*)(int, int))dlsym(RTLD_NEXT, "flock");
    }
    const char *d = getenv("SHIM_FLOCK_DELAY_MS");
    if (d && (op & LOCK_EX)) {
        char p[64], path[4096];
        snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(p, path, sizeof path - 1);
        if (n > 0) {
            path[n] = 0;
            if (own(path)) {
                fprintf(stderr, "shim: flock(%s) delayed %s ms\n", path, d);
                nap(d);
            }
        }
    }
    return real(fd, op);
}

/* Set once this process's rename of its `.lock.new` was reported lost. */
static int lost;

int rename(const char *old, const char *new) {
    static int (*real)(const char *, const char *);
    if (!real) {
        real = (int (*)(const char *, const char *))dlsym(RTLD_NEXT, "rename");
    }
    int r = real(old, new);
    if (r == 0 && getenv("SHIM_RENAME_LOST_MS") && own(old) && is_new(old)) {
        lost = 1;
        errno = ENOENT;
        return -1;
    }
    return r;
}

int unlink(const char *path) {
    static int (*real)(const char *);
    if (!real) {
        real = (int (*)(const char *))dlsym(RTLD_NEXT, "unlink");
    }
    int r = real(path);
    int e = errno;
    const char *d = getenv("SHIM_RENAME_LOST_MS");
    if (d && lost && own(path) && is_new(path)) {
        fprintf(stderr, "shim: rename(%s) reported lost; unlink delayed %s ms\n", path, d);
        nap(d);
    }
    errno = e;
    return r;
}

DIR *opendir(const char *path) {
    static DIR *(*real)(const char *);
    if (!real) {
        real = (DIR *(*)(const char *))dlsym(RTLD_NEXT, "opendir");
    }
    const char *v = getenv("SHIM_VANISH_DIR");
    const char *last = path ? strrchr(path, '/') : NULL;
    if (v && last && strcmp(last + 1, v) == 0 && rmdir(path) == 0) {
        fprintf(stderr, "shim: opendir(%s) removed first\n", path);
    }
    return real(path);
}
