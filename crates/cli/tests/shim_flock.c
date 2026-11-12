/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* LD_PRELOAD shim for outdir_race.rs (a_run_taking_its_lock_keeps_its_directory): delays this
 * process's flock(LOCK_EX) on the lock file of its own run directory (a path holding "-<pid>-",
 * the pid /proc/self names, whose last component starts with ".lock") by SHIM_FLOCK_DELAY_MS
 * milliseconds, and says so on stderr ("shim: flock(<path>) delayed <ms> ms"). It widens the
 * time between a run directory's creation and its lock. Every other flock goes through
 * unchanged.
 *
 *   cc -shared -fPIC -o shim_flock.so shim_flock.c -ldl */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

int flock(int fd, int op) {
    static int (*real)(int, int);
    if (!real) {
        real = (int (*)(int, int))dlsym(RTLD_NEXT, "flock");
    }
    const char *d = getenv("SHIM_FLOCK_DELAY_MS");
    if (d && (op & LOCK_EX)) {
        char p[64], path[4096], self[64], me[80];
        snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(p, path, sizeof path - 1);
        ssize_t m = readlink("/proc/self", self, sizeof self - 1);
        if (n > 0 && m > 0) {
            path[n] = 0;
            self[m] = 0;
            snprintf(me, sizeof me, "-%s-", self);
            const char *last = strrchr(path, '/');
            if (strstr(path, me) && last && strncmp(last, "/.lock", 6) == 0) {
                fprintf(stderr, "shim: flock(%s) delayed %s ms\n", path, d);
                int ms = atoi(d);
                struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
                nanosleep(&ts, NULL);
            }
        }
    }
    return real(fd, op);
}
