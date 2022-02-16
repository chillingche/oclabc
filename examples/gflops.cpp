#include <sys/syscall.h>
#include <sys/time.h>
#include <sched.h>
#include <stdint.h>
#include <unistd.h>
#include "log.h"

static int set_thread_affinity(int num)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(num, &mask);
    int status = syscall(__NR_sched_setaffinity, gettid(), sizeof(mask), &mask);
    if (status) {
        LOGE("fail to set affinity %d", status);
        return -1;
    }
    LOGI("bind core %d", num);
    return 0;
}

static void set_thread_affinity_big() {
    set_thread_affinity(7);
}

static void set_thread_affinity_mid() {
    set_thread_affinity(6);
    set_thread_affinity(5);
    set_thread_affinity(4);
}

static void set_thread_affinity_small() {
    set_thread_affinity(3);
    set_thread_affinity(2);
    set_thread_affinity(1);
    set_thread_affinity(0);
}

void mvm_row_kernel(uint32_t N, uint32_t K, float *matrix, float *vector, float *result)
{
    int KTail = K % 4;
    int KInner = K - KTail;
    float *w0 = matrix;
    float *w1 = matrix + K * N;
    float *w2 = matrix + K * 2 * N;
    float *w3 = matrix + K * 3 * N;
    asm volatile("mov x19, %5\n"
                 "ld1 {v18.s}[0], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.s}[1], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.s}[2], [x19]\n"
                 "add x19, x19, %8\n"
                 "ld1 {v18.s}[3], [x19]\n"

                 "movi v17.4s, #0x0\n"
                 "movi v16.4s, #0x0\n"
                 "movi v9.4s, #0x0\n"
                 "movi v10.4s, #0x0\n"
                 "movi v11.4s, #0x0\n"
                 "movi v12.4s, #0x0\n"
                 "mov x20, %6\n"
                 "cmp x20, #0x0\n"
                 "beq 3f\n"
                 "0:\n"

                 "ld1 {v0.4s}, [%0], #16\n"
                 "ld1 {v1.4s}, [%1], #16\n"
                 "ld1 {v2.4s}, [%2], #16\n"
                 "ld1 {v3.4s}, [%3], #16\n"
                 "ld1 {v4.4s}, [%4], #16\n"

                 "fmla v9.4s,  v1.4s, v0.4s\n"
                 "fmla v10.4s,  v2.4s, v0.4s\n"
                 "fmla v11.4s,  v3.4s, v0.4s\n"
                 "fmla v12.4s,  v4.4s, v0.4s\n"

                 "subs x20, x20, #4\n"
                 "bne 0b\n"

                 "faddp v13.4s,  v9.4s, v10.4s\n"
                 "faddp v14.4s, v11.4s, v12.4s\n"
                 "faddp v17.4s, v13.4s, v14.4s\n"
                 "3:\n"
                 "mov x16, %7\n"
                 "cmp x16, #0x0\n"
                 "beq 2f\n"

                 "1:\n"
                 "ld1 {v8.s}[0], [%0], #4\n"

                 "ld1 {v1.s}[0], [%1], #4\n"
                 "ld1 {v1.s}[1], [%2], #4\n"
                 "ld1 {v1.s}[2], [%3], #4\n"
                 "ld1 {v1.s}[3], [%4], #4\n"
                 "fmla v16.4s,  v1.4s, v8.s[0]\n"

                 "subs x16, x16, 0x1\n"
                 "bne 1b\n"

                 "fadd v17.4s, v17.4s, v16.4s\n"

                 "2:\n"

                 "fadd v17.4s, v17.4s, v18.4s\n"
                 "mov x19, %5\n"
                 "st1 {v17.s}[0], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.s}[1], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.s}[2], [x19]\n"
                 "add x19, x19, %8\n"
                 "st1 {v17.s}[3], [x19]\n"
                 : "+r"(vector), "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3), "+r"(result)
                 : "r"((int64_t)KInner), "r"((int64_t)KTail), "r"((int64_t)N * 4)
                 : "memory", "cc", "x19", "x20", "x21", "x22", "x23", "x24", "x15", "x16", "v0",
                 "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16",
                 "v17", "v18");
}

static void freefma_a53(int64_t loop, int64_t *num_mac)
{
    asm volatile(
        "mov x9, %0\n"
        "0:\n"
        "fmla v0.4s,  v12.4s, v13.s[0]\n"
        "fmla v1.4s,  v12.4s, v13.s[1]\n"
        "fmla v2.4s,  v12.4s, v13.s[2]\n"
        "fmla v3.4s,  v12.4s, v13.s[3]\n"
        "subs x9, x9, #1\n"
        "fmla v4.4s,  v12.4s, v13.s[0]\n"
        "bne 0b\n"
        :
        : "r"(loop)
        : "x9", "v0", "v1", "v2", "v3", "v4",
          "v12", "v13");
    *num_mac = 5 * 4;
}

static void freefma_f16_a55(int64_t loop, int64_t *num_mac)
{
    asm volatile(
        "mov x9, %0\n"
        "0:\n"
        "fmla v0.8h,  v12.8h, v13.h[0]\n"
        "fmla v1.8h,  v12.8h, v13.h[1]\n"
        "fmla v2.8h,  v12.8h, v13.h[2]\n"
        "fmla v3.8h,  v12.8h, v13.h[3]\n"
        "subs x9, x9, #1\n"
        "fmla v4.8h,  v12.8h, v13.h[4]\n"
        "bne 0b\n"
        :
        : "r"(loop)
        : "x9", "v0", "v1", "v2", "v3", "v4", "v12", "v13");
    *num_mac = 5 * 8;
}

static void freefma_a76(int64_t loop, int64_t *num_mac)
{
    asm volatile(
        "mov x9, %0\n"
        "0:\n"
        "fmla v0.4s,  v12.4s, v13.s[0]\n"
        "fmla v1.4s,  v12.4s, v13.s[1]\n"
        "fmla v2.4s,  v12.4s, v13.s[2]\n"
        "fmla v3.4s,  v12.4s, v13.s[3]\n"
        "subs x9, x9, #1\n"
        "fmla v4.4s,  v12.4s, v13.s[0]\n"
        "fmla v5.4s,  v12.4s, v13.s[1]\n"
        "fmla v6.4s,  v12.4s, v13.s[2]\n"
        "fmla v7.4s,  v12.4s, v13.s[3]\n"
        "bne 0b\n"
        :
        : "r"(loop)
        : "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v12", "v13");
    *num_mac = 8 * 4;
}

static void freefma_x1(int64_t loop, int64_t *num_mac)
{
    asm volatile(
        "mov x9, %0\n"
        "0:\n"
        "fmla v0.4s,  v12.4s, v13.s[0]\n"
        "fmla v1.4s,  v12.4s, v13.s[1]\n"
        "fmla v2.4s,  v12.4s, v13.s[2]\n"
        "fmla v3.4s,  v12.4s, v13.s[3]\n"

        "fmla v4.4s,  v12.4s, v13.s[0]\n"
        "fmla v5.4s,  v12.4s, v13.s[1]\n"
        "fmla v6.4s,  v12.4s, v13.s[2]\n"
        "fmla v7.4s,  v12.4s, v13.s[3]\n"

        "subs x9, x9, #1\n"

        "fmla v8.4s,  v12.4s, v13.s[0]\n"
        "fmla v9.4s,  v12.4s, v13.s[1]\n"
        "fmla v10.4s,  v12.4s, v13.s[2]\n"
        "fmla v11.4s,  v12.4s, v13.s[3]\n"

        "fmla v14.4s,  v12.4s, v13.s[0]\n"
        "fmla v15.4s,  v12.4s, v13.s[1]\n"
        "fmla v16.4s,  v12.4s, v13.s[2]\n"
        "fmla v17.4s,  v12.4s, v13.s[3]\n"
        "bne 0b\n"
        :
        : "r"(loop)
        : "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v12", "v13", "v14", "v15", "v16", "v17");
    *num_mac = 16 * 4;
}

int main(int argc, char const *argv[])
{
    int64_t loop = 1024 * 1024 * 1024;
    int64_t warmup = 1024 * 1024;
    {
        set_thread_affinity_small();
        int64_t macs = 0;
        freefma_a53(warmup, &macs);
        timeval begin, end;
        gettimeofday(&begin, NULL);
        freefma_a53(loop, &macs);
        gettimeofday(&end, NULL);
        float lt = (end.tv_sec - begin.tv_sec) * 1000.0f +
                   (end.tv_usec - begin.tv_usec) / 1000.0f;
        float gfp = loop * macs * 2.0f / lt * 1000.0f / 1000.0f /
                    1000.0f / 1000.0f;
        LOGI("time elapsed: %.4f ms", lt);
        LOGI("F32 GFLOPS: %.4f", gfp);
    }

    {
        set_thread_affinity_small();
        int64_t macs = 0;
        freefma_f16_a55(warmup, &macs);
        timeval begin, end;
        gettimeofday(&begin, NULL);
        freefma_f16_a55(loop, &macs);
        gettimeofday(&end, NULL);
        float lt = (end.tv_sec - begin.tv_sec) * 1000.0f +
                   (end.tv_usec - begin.tv_usec) / 1000.0f;
        float gfp = loop * macs * 2.0f / lt * 1000.0f / 1000.0f /
                    1000.0f / 1000.0f;
        LOGI("time elapsed: %.4f ms", lt);
        LOGI("F16 GFLOPS: %.4f", gfp);
    }

    {
        set_thread_affinity_mid();
        int64_t macs = 0;
        freefma_a76(warmup, &macs);
        timeval begin, end;
        gettimeofday(&begin, NULL);
        freefma_a76(loop, &macs);
        gettimeofday(&end, NULL);
        float lt = (end.tv_sec - begin.tv_sec) * 1000.0f +
                   (end.tv_usec - begin.tv_usec) / 1000.0f;
        float gfp = loop * macs * 2.0f / lt * 1000.0f / 1000.0f /
                    1000.0f / 1000.0f;
        LOGI("time elapsed: %.4f ms", lt);
        LOGI("F32 GFLOPS: %.4f", gfp);
    }

    {
        set_thread_affinity_big();
        int64_t macs = 0;
        freefma_a76(warmup, &macs);
        timeval begin, end;
        gettimeofday(&begin, NULL);
        freefma_a76(loop, &macs);
        gettimeofday(&end, NULL);
        float lt = (end.tv_sec - begin.tv_sec) * 1000.0f +
                   (end.tv_usec - begin.tv_usec) / 1000.0f;
        float gfp = loop * macs * 2.0f / lt * 1000.0f / 1000.0f /
                    1000.0f / 1000.0f;
        LOGI("time elapsed: %.4f ms", lt);
        LOGI("F32 GFLOPS: %.4f", gfp);
    }

    return 0;
}

