/*
 * tb_cosim.cpp – File-driven co-simulation testbench for yolov2_accel
 *
 * For each layer directory under hls/tb/cosim_data/<N>/ :
 *   1. Read meta.txt  (layer dimensions, command)
 *   2. Load input.bin, weight.bin → emulated DDR
 *   3. Call yolov2_accel() with the appropriate command
 *   4. Read output from emulated DDR
 *   5. Compare against expected.bin
 *   6. Write output to hls_output.bin  (for Python comparison)
 *
 * Run from Vivado HLS cosim_design step; the test vectors are generated
 * by  tools/verify/generate_cosim_vectors.py.
 *
 * Compile (C simulation only):
 *   g++ -std=c++14 -I../src tb_cosim.cpp -o tb_cosim && ./tb_cosim
 *
 * Note: In C simulation the AXI pragma macros are no-ops; the test still
 *       exercises the functional behaviour of every engine.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include "../src/yolov2_accel.h"
#include "cosim_utils.h"

// ============================================================
// Configuration
// ============================================================

// Adjust this path if running from a different working directory.
#define COSIM_DATA_DIR  "hls/tb/cosim_data"

// DDR byte addresses used by the testbench
#define TB_IN_ADDR   0x00000000u
#define TB_OUT_ADDR  0x02000000u   // 32 MB above input
#define TB_WT_ADDR   0x04000000u   // 64 MB above base

// ============================================================
// Global DDR emulation array
// ============================================================

static axi_word_t ddr[DDR_WORDS];

// ============================================================
// Run one layer test
// ============================================================

static int run_layer_test(const char *layer_dir)
{
    char meta_path[512], in_path[512], wt_path[512], exp_path[512], out_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.txt",      layer_dir);
    snprintf(in_path,   sizeof(in_path),   "%s/input.bin",     layer_dir);
    snprintf(wt_path,   sizeof(wt_path),   "%s/weight.bin",    layer_dir);
    snprintf(exp_path,  sizeof(exp_path),  "%s/expected.bin",  layer_dir);
    snprintf(out_path,  sizeof(out_path),  "%s/hls_output.bin",layer_dir);

    // ---- Load metadata
    LayerMeta m;
    if (!load_meta(meta_path, m)) {
        fprintf(stderr, "[cosim] SKIP %s (no meta.txt)\n", layer_dir);
        return 0;
    }

    unsigned n_in  = m.in_c * m.in_h * m.in_w;
    unsigned n_out = m.out_c * m.out_h * m.out_w;
    unsigned n_wt  = 0;
    if (!strcmp(m.cmd, "CONV"))
        n_wt = m.out_c * m.in_c * m.ksize * m.ksize + m.out_c; // weights + biases

    printf("\n[layer %s]  cmd=%-5s  in=%u×%u×%u  out=%u×%u×%u  "
           "k=%u s=%u p=%u\n",
           layer_dir + strlen(COSIM_DATA_DIR) + 1,
           m.cmd, m.in_c, m.in_h, m.in_w,
           m.out_c, m.out_h, m.out_w,
           m.ksize, m.stride, m.pad);

    // ---- Allocate local buffers
    data_t *in_buf  = new data_t[n_in];
    data_t *out_buf = new data_t[n_out];
    data_t *exp_buf = new data_t[n_out];
    data_t *wt_buf  = (n_wt > 0) ? new data_t[n_wt] : nullptr;

    // ---- Load test vectors
    if (!load_bin(in_path,  in_buf,  n_in))  goto fail;
    if (!load_bin(exp_path, exp_buf, n_out)) goto fail;
    if (n_wt > 0 && !load_bin(wt_path, wt_buf, n_wt)) goto fail;

    // ---- Write to emulated DDR
    memset(ddr, 0, sizeof(ddr));
    ddr_write_array(ddr, TB_IN_ADDR, in_buf,  n_in);
    if (wt_buf)
        ddr_write_array(ddr, TB_WT_ADDR, wt_buf, n_wt);

    // ---- Determine command code
    {
        unsigned cmd = CMD_IDLE;
        if      (!strcmp(m.cmd, "CONV"))  cmd = CMD_CONV;
        else if (!strcmp(m.cmd, "POOL"))  cmd = CMD_POOL;
        else if (!strcmp(m.cmd, "REORG")) cmd = CMD_REORG;
        else {
            fprintf(stderr, "[cosim]   SKIP unknown cmd=%s\n", m.cmd);
            goto done;
        }

        // ---- Call the accelerator
        yolov2_accel(
            ddr,
            cmd,
            TB_IN_ADDR,
            TB_OUT_ADDR,
            TB_WT_ADDR,
            m.in_h,   m.in_w,   m.in_c,
            m.out_c,
            m.ksize,  m.stride, m.pad,
            m.flags
        );
    }

    // ---- Read back output from emulated DDR
    ddr_read_array(ddr, TB_OUT_ADDR, out_buf, n_out);

    // ---- Compare
    {
        char label[64];
        snprintf(label, sizeof(label), "%s %s",
                 layer_dir + strlen(COSIM_DATA_DIR) + 1, m.cmd);
        CmpStats s = compare_outputs(exp_buf, out_buf, n_out);
        print_stats(label, s);

        // Save output for Python post-processing
        save_bin(out_path, out_buf, n_out);

        if (s.snr_db < 15.0) {
            fprintf(stderr, "[cosim]   WARNING: SNR %.1f dB < 15 dB threshold\n",
                    s.snr_db);
        }
    }

done:
    delete[] in_buf;
    delete[] out_buf;
    delete[] exp_buf;
    if (wt_buf) delete[] wt_buf;
    return 0;

fail:
    fprintf(stderr, "[cosim]   FAIL: could not load test vectors for %s\n", layer_dir);
    delete[] in_buf;
    delete[] out_buf;
    delete[] exp_buf;
    if (wt_buf) delete[] wt_buf;
    return 1;
}

// ============================================================
// Self-contained unit tests (no file I/O, for C-sim sanity)
// ============================================================

static int unit_test_conv()
{
    printf("\n[unit] CONV 3×3 id-weight 4-ch 8×8 ...\n");

    const unsigned IH=8, IW=8, IC=4, OC=4, KS=3;
    const unsigned N_IN  = IC * IH * IW;
    const unsigned N_OUT = OC * IH * IW;  // stride=1, pad=1 → same size
    const unsigned N_WT  = OC * IC * KS * KS + OC;

    memset(ddr, 0, sizeof(ddr));

    // Input: channel c, row r, col c → value = c + r*0.1 + c*0.01
    {
        data_t *buf = new data_t[N_IN];
        for (unsigned c=0; c<IC; c++)
            for (unsigned r=0; r<IH; r++)
                for (unsigned col=0; col<IW; col++)
                    buf[c*IH*IW + r*IW + col] =
                        data_t((float)c + r*0.1f + col*0.01f);
        ddr_write_array(ddr, TB_IN_ADDR, buf, N_IN);
        delete[] buf;
    }

    // Weights: identity (centre kh=1,kw=1 → 1 for oc==ic, else 0)
    {
        data_t *wbuf = new data_t[N_WT];
        for (unsigned i=0; i<N_WT; i++) wbuf[i] = data_t(0);
        for (unsigned oc=0; oc<OC; oc++) {
            unsigned idx = oc * IC * KS * KS + oc * KS * KS + 1*KS + 1;
            wbuf[idx] = data_t(1);
        }
        ddr_write_array(ddr, TB_WT_ADDR, wbuf, N_WT);
        delete[] wbuf;
    }

    yolov2_accel(ddr, CMD_CONV,
                 TB_IN_ADDR, TB_OUT_ADDR, TB_WT_ADDR,
                 IH, IW, IC, OC, KS, 1, 1,
                 FLAG_LRELU);

    data_t *out = new data_t[N_OUT];
    ddr_read_array(ddr, TB_OUT_ADDR, out, N_OUT);

    int errs = 0;
    for (unsigned oc=0; oc<OC; oc++)
        for (unsigned r=1; r<IH-1; r++)   // skip edge (zero-pad affected)
            for (unsigned c=1; c<IW-1; c++) {
                float exp_v = (float)oc + r*0.1f + c*0.01f;
                float got   = (float)out[oc*IH*IW + r*IW + c];
                float diff  = fabsf(got - exp_v);
                if (diff > 0.05f) { errs++; }
            }

    delete[] out;
    printf("  %s (%d errors in interior pixels)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

static int unit_test_pool()
{
    printf("\n[unit] POOL 2×2 stride=2 8×8 ch=2 ...\n");
    const unsigned IH=8, IW=8, IC=2;
    const unsigned OH=4, OW=4;
    const unsigned N_IN=IC*IH*IW, N_OUT=IC*OH*OW;

    memset(ddr, 0, sizeof(ddr));

    data_t *ibuf = new data_t[N_IN];
    for (unsigned c=0; c<IC; c++)
        for (unsigned r=0; r<IH; r++)
            for (unsigned col=0; col<IW; col++)
                ibuf[c*IH*IW + r*IW + col] = data_t(c*10.0f + r + col/8.0f);
    ddr_write_array(ddr, TB_IN_ADDR, ibuf, N_IN);
    delete[] ibuf;

    yolov2_accel(ddr, CMD_POOL, TB_IN_ADDR, TB_OUT_ADDR, 0,
                 IH, IW, IC, IC, 2, 2, 0, 0);

    data_t *out = new data_t[N_OUT];
    ddr_read_array(ddr, TB_OUT_ADDR, out, N_OUT);

    int errs = 0;
    for (unsigned c=0; c<IC; c++)
        for (unsigned r=0; r<OH; r++)
            for (unsigned col=0; col<OW; col++) {
                float mx = -1e9;
                for (int kr=0; kr<2; kr++) for (int kc=0; kc<2; kc++) {
                    float v = c*10.0f + (r*2+kr) + (col*2+kc)/8.0f;
                    if (v > mx) mx = v;
                }
                float diff = fabsf((float)out[c*OH*OW + r*OW + col] - mx);
                if (diff > 0.05f) errs++;
            }

    delete[] out;
    printf("  %s (%d errors)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

static int unit_test_reorg()
{
    printf("\n[unit] REORG stride=2 4×4×2 → 2×2×8 ...\n");
    const unsigned IH=4, IW=4, IC=2, STR=2;
    const unsigned OH=2, OW=2, OC=8;
    const unsigned N_IN=IC*IH*IW, N_OUT=OC*OH*OW;

    memset(ddr, 0, sizeof(ddr));

    data_t *ibuf = new data_t[N_IN];
    for (unsigned c=0; c<IC; c++)
        for (unsigned r=0; r<IH; r++)
            for (unsigned col=0; col<IW; col++)
                ibuf[c*IH*IW + r*IW + col] = data_t(c*100.0f + r*10.0f + col);
    ddr_write_array(ddr, TB_IN_ADDR, ibuf, N_IN);
    delete[] ibuf;

    yolov2_accel(ddr, CMD_REORG, TB_IN_ADDR, TB_OUT_ADDR, 0,
                 IH, IW, IC, OC, 2, STR, 0, 0);

    data_t *out = new data_t[N_OUT];
    ddr_read_array(ddr, TB_OUT_ADDR, out, N_OUT);

    int errs = 0;
    for (unsigned ic=0; ic<IC; ic++)
        for (unsigned kh=0; kh<STR; kh++)
            for (unsigned kw=0; kw<STR; kw++) {
                unsigned oc = ic*STR*STR + kh*STR + kw;
                for (unsigned r=0; r<OH; r++)
                    for (unsigned col=0; col<OW; col++) {
                        float exp_v = ic*100.0f + (r*2+kh)*10.0f + (col*2+kw);
                        float got   = (float)out[oc*OH*OW + r*OW + col];
                        if (fabsf(got - exp_v) > 0.05f) errs++;
                    }
            }

    delete[] out;
    printf("  %s (%d errors)\n", errs==0?"PASS":"FAIL", errs);
    return errs;
}

// ============================================================
// main
// ============================================================

int main(int argc, char **argv)
{
    printf("============================================================\n");
    printf("  YOLOv2 HLS Co-simulation Testbench\n");
    printf("============================================================\n");

    int total_errs = 0;

    // ---- Unit tests (no file I/O, always run) ----------------------------
    printf("\n--- Unit tests (built-in synthetic data) ---\n");
    total_errs += unit_test_conv();
    total_errs += unit_test_pool();
    total_errs += unit_test_reorg();

    // ---- File-driven tests (requires cosim_data/ from Python generator) --
    const char *data_dir = (argc > 1) ? argv[1] : COSIM_DATA_DIR;

    DIR *dir = opendir(data_dir);
    if (!dir) {
        printf("\n[cosim] No test vector directory found at '%s'.\n", data_dir);
        printf("        Run:  python3 tools/verify/run_verification.py\n");
        printf("        Then re-run this testbench.\n");
    } else {
        printf("\n--- File-driven tests (directory: %s) ---\n", data_dir);
        printf("  %-28s  %8s  %10s  %10s  %8s\n",
               "Layer", "MAE", "MaxErr", "SNR(dB)", "%≤1LSB");
        printf("  %s\n", std::string(70, '-').c_str());

        struct dirent *ent;
        std::vector<std::string> entries;
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            char full[512];
            snprintf(full, sizeof(full), "%s/%s", data_dir, ent->d_name);
            struct stat st;
            if (stat(full, &st) == 0 && S_ISDIR(st.st_mode))
                entries.push_back(full);
        }
        closedir(dir);
        std::sort(entries.begin(), entries.end(),
            [](const std::string &a, const std::string &b) {
                return atoi(a.c_str() + a.rfind('/')+1) <
                       atoi(b.c_str() + b.rfind('/')+1);
            });

        for (auto &e : entries)
            total_errs += run_layer_test(e.c_str());
    }

    // ---- Summary ----------------------------------------------------------
    printf("\n%s\n", std::string(60, '=').c_str());
    printf("  Testbench %s  (total anomalies: %d)\n",
           total_errs == 0 ? "PASSED" : "WARNINGS", total_errs);
    printf("%s\n\n", std::string(60, '=').c_str());
    return (total_errs > 0) ? 1 : 0;
}
