#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include "../src/defines.h"

// ============================================================
// Binary file I/O  (int16, little-endian, matches quantize_utils.py)
// ============================================================

static bool load_bin(const char *path, data_t *buf, unsigned n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[cosim] ERROR: cannot open %s\n", path);
        return false;
    }
    for (unsigned i = 0; i < n; i++) {
        int16_t raw;
        if (fread(&raw, 2, 1, fp) != 1) { fclose(fp); return false; }
        buf[i].range(15, 0) = (ap_uint<16>)(uint16_t)raw;
    }
    fclose(fp);
    return true;
}

static bool save_bin(const char *path, const data_t *buf, unsigned n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[cosim] ERROR: cannot write %s\n", path);
        return false;
    }
    for (unsigned i = 0; i < n; i++) {
        int16_t raw = (int16_t)(uint16_t)buf[i].range(15, 0);
        fwrite(&raw, 2, 1, fp);
    }
    fclose(fp);
    return true;
}

// ============================================================
// Metadata reader (key=value text file)
// ============================================================

struct LayerMeta {
    char     cmd[8];
    unsigned in_h, in_w, in_c;
    unsigned out_c, out_h, out_w;
    unsigned ksize, stride, pad, flags;
};

static bool load_meta(const char *path, LayerMeta &m) {
    FILE *fp = fopen(path, "r");
    if (!fp) return false;
    memset(&m, 0, sizeof(m));
    char key[32], val[32];
    while (fscanf(fp, "%31[^=]=%31s\n", key, val) == 2) {
        if (!strcmp(key,"cmd"))    { strncpy(m.cmd, val, 7); }
        else if (!strcmp(key,"in_h"))   m.in_h   = atoi(val);
        else if (!strcmp(key,"in_w"))   m.in_w   = atoi(val);
        else if (!strcmp(key,"in_c"))   m.in_c   = atoi(val);
        else if (!strcmp(key,"out_c"))  m.out_c  = atoi(val);
        else if (!strcmp(key,"out_h"))  m.out_h  = atoi(val);
        else if (!strcmp(key,"out_w"))  m.out_w  = atoi(val);
        else if (!strcmp(key,"ksize"))  m.ksize  = atoi(val);
        else if (!strcmp(key,"stride")) m.stride = atoi(val);
        else if (!strcmp(key,"pad"))    m.pad    = atoi(val);
        else if (!strcmp(key,"flags"))  m.flags  = atoi(val);
    }
    fclose(fp);
    return true;
}

// ============================================================
// Comparison statistics
// ============================================================

struct CmpStats {
    double mae;
    double max_err;
    double snr_db;
    double pct_lsb;   // % within 1 LSB
    unsigned n;
};

static CmpStats compare_outputs(const data_t *ref, const data_t *dut, unsigned n) {
    CmpStats s = {0};
    s.n = n;
    double sig = 0, noise = 0;
    for (unsigned i = 0; i < n; i++) {
        double r = (double)(float)ref[i];
        double d = (double)(float)dut[i];
        double e = fabs(r - d);
        s.mae     += e;
        s.max_err  = (e > s.max_err) ? e : s.max_err;
        sig       += r * r;
        noise     += (r - d) * (r - d);
        if (e <= 1.0 / 256.0) s.pct_lsb += 1;
    }
    s.mae     /= n;
    s.pct_lsb  = s.pct_lsb / n * 100.0;
    s.snr_db   = (noise < 1e-12) ? 999.0 : 10.0 * log10(sig / noise);
    return s;
}

static void print_stats(const char *label, const CmpStats &s) {
    printf("  %-28s  MAE=%8.4f  MaxErr=%8.4f  SNR=%6.1f dB  %%≤1LSB=%5.1f%%  %s\n",
           label, s.mae, s.max_err, s.snr_db, s.pct_lsb,
           (s.snr_db >= 20.0) ? "PASS" : "WARN");
}

// ============================================================
// DDR emulation helpers (flat array, byte-addressed)
// ============================================================

#define DDR_MB    64
#define DDR_BYTES (DDR_MB * 1024 * 1024)
#define DDR_WORDS (DDR_BYTES / 8)

static void ddr_write_array(axi_word_t *ddr, unsigned byte_off,
                              const data_t *buf, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        unsigned off  = byte_off + i * 2;
        unsigned wo   = off >> 3;
        int      lane = (off & 7) >> 1;
        axi_word_t w  = ddr[wo];
        ap_uint<16> bits;
        bits.range(15,0) = buf[i].range(15,0);
        w.range(16*lane+15, 16*lane) = bits;
        ddr[wo] = w;
    }
}

static void ddr_read_array(const axi_word_t *ddr, unsigned byte_off,
                             data_t *buf, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        unsigned off  = byte_off + i * 2;
        unsigned wo   = off >> 3;
        int      lane = (off & 7) >> 1;
        ap_uint<16> bits = ddr[wo].range(16*lane+15, 16*lane);
        buf[i].range(15,0) = bits.range(15,0);
    }
}
