#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <riscv_vector.h>
#include <utility>

#define WARM_UP_COUNT 10
#define TEST_COUNT 50

typedef void (*method_t)(size_t M, size_t K, size_t N, const float *A,
                         const float *B, const float *Bias, float *C, float min,
                         float max);

void method_ref(size_t M, size_t K, size_t N, const float *A, const float *B,
                const float *Bias, float *C, float min, float max) {
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      float acc = Bias[n];
      for (size_t k = 0; k < K; k++) {
        acc += A[m * K + k] * B[k * N + n];
      }
      acc = std::max(std::min(acc, max), min);
      *C++ = acc;
    }
  }
}

void method_1(size_t M, size_t K, size_t N, const float *A, const float *B,
              const float *Bias, float *C, float min, float max) {
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  for (size_t m = 0; m < M; m++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t n = N; n; n -= vl) {
      vl = vsetvl_e32m8(n);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc = vle32_v_f32m8(pbias, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m8(pb_vl, vl);
        acc = vfmacc_vf_f32m8(acc, *pa++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc = vfmax_vf_f32m8(vfmin_vf_f32m8(acc, max, vl), min, vl);

      vse32_v_f32m8(pc, acc, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }
    ptr_a += K;
    ptr_out += N;
  }
}

void method_2(size_t M, size_t K, size_t N, const float *A, const float *B,
              const float *Bias, float *C, float min, float max) {
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  for (size_t m = 0; m < M; m++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t n = N; n; n -= vl) {
      vl = vsetvl_e32m1(n);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc = vle32_v_f32m1(pbias, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m1(pb_vl, vl);
        acc = vfmacc_vf_f32m1(acc, *pa++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc = vfmax_vf_f32m1(vfmin_vf_f32m1(acc, max, vl), min, vl);

      vse32_v_f32m1(pc, acc, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }
    ptr_a += K;
    ptr_out += N;
  }
}

void method_3(size_t M, size_t K, size_t N, const float *A, const float *B,
              const float *Bias, float *C, float min, float max) {
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  auto NN = N / 4;
  auto REMAIN = N - NN * 4;
  for (size_t m = 0; m < M; m++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t nn = NN; nn; nn -= vl) {
      vl = vsetvl_e32m4(nn);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc0 = vle32_v_f32m4(pbias + vl * 0, vl);
      auto acc1 = vle32_v_f32m4(pbias + vl * 1, vl);
      auto acc2 = vle32_v_f32m4(pbias + vl * 2, vl);
      auto acc3 = vle32_v_f32m4(pbias + vl * 3, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb0 = vle32_v_f32m4(pb_vl + vl * 0, vl);
        auto vb1 = vle32_v_f32m4(pb_vl + vl * 1, vl);
        auto vb2 = vle32_v_f32m4(pb_vl + vl * 2, vl);
        auto vb3 = vle32_v_f32m4(pb_vl + vl * 3, vl);

        acc0 = vfmacc_vf_f32m4(acc0, *pa, vb0, vl);
        acc1 = vfmacc_vf_f32m4(acc1, *pa, vb1, vl);
        acc2 = vfmacc_vf_f32m4(acc2, *pa, vb2, vl);
        acc3 = vfmacc_vf_f32m4(acc3, *pa, vb3, vl);
        pb_vl += N;
        pa++;
      }

      // update acc with act
      acc0 = vfmin_vf_f32m4(acc0, max, vl);
      acc1 = vfmin_vf_f32m4(acc1, max, vl);
      acc2 = vfmin_vf_f32m4(acc2, max, vl);
      acc3 = vfmin_vf_f32m4(acc3, max, vl);

      acc0 = vfmax_vf_f32m4(acc0, min, vl);
      acc1 = vfmax_vf_f32m4(acc1, min, vl);
      acc2 = vfmax_vf_f32m4(acc2, min, vl);
      acc3 = vfmax_vf_f32m4(acc3, min, vl);

      vse32_v_f32m4(pc + vl * 0, acc0, vl);
      vse32_v_f32m4(pc + vl * 1, acc1, vl);
      vse32_v_f32m4(pc + vl * 2, acc2, vl);
      vse32_v_f32m4(pc + vl * 3, acc3, vl);

      pb += vl;
      pc += vl;
      pbias += vl;
    }

    vl = 0;
    pb = B + NN * 4;
    pc = ptr_out + NN * 4;
    pbias = Bias + NN * 4;
    for (size_t n = REMAIN; n; n -= vl) {
      vl = vsetvl_e32m8(n);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc = vle32_v_f32m8(pbias, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m8(pb_vl, vl);
        acc = vfmacc_vf_f32m8(acc, *pa++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc = vfmax_vf_f32m8(vfmin_vf_f32m8(acc, max, vl), min, vl);

      vse32_v_f32m8(pc, acc, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }

    ptr_a += K;
    ptr_out += N;
  }
}

void method_4(size_t M, size_t K, size_t N, const float *A, const float *B,
              const float *Bias, float *C, float min, float max) {
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  auto NN = N / 8;
  auto REMAIN = N - NN * 8;
  for (size_t m = 0; m < M; m++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t nn = NN; nn; nn -= vl) {
      vl = vsetvl_e32m1(nn);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc0 = vle32_v_f32m1(pbias + vl * 0, vl);
      auto acc1 = vle32_v_f32m1(pbias + vl * 1, vl);
      auto acc2 = vle32_v_f32m1(pbias + vl * 2, vl);
      auto acc3 = vle32_v_f32m1(pbias + vl * 3, vl);
      auto acc4 = vle32_v_f32m1(pbias + vl * 4, vl);
      auto acc5 = vle32_v_f32m1(pbias + vl * 5, vl);
      auto acc6 = vle32_v_f32m1(pbias + vl * 6, vl);
      auto acc7 = vle32_v_f32m1(pbias + vl * 7, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb0 = vle32_v_f32m1(pb_vl + vl * 0, vl);
        auto vb1 = vle32_v_f32m1(pb_vl + vl * 1, vl);
        auto vb2 = vle32_v_f32m1(pb_vl + vl * 2, vl);
        auto vb3 = vle32_v_f32m1(pb_vl + vl * 3, vl);
        auto vb4 = vle32_v_f32m1(pb_vl + vl * 4, vl);
        auto vb5 = vle32_v_f32m1(pb_vl + vl * 5, vl);
        auto vb6 = vle32_v_f32m1(pb_vl + vl * 6, vl);
        auto vb7 = vle32_v_f32m1(pb_vl + vl * 7, vl);

        acc0 = vfmacc_vf_f32m1(acc0, *pa, vb0, vl);
        acc1 = vfmacc_vf_f32m1(acc1, *pa, vb1, vl);
        acc2 = vfmacc_vf_f32m1(acc2, *pa, vb2, vl);
        acc3 = vfmacc_vf_f32m1(acc3, *pa, vb3, vl);
        acc4 = vfmacc_vf_f32m1(acc0, *pa, vb4, vl);
        acc5 = vfmacc_vf_f32m1(acc1, *pa, vb5, vl);
        acc6 = vfmacc_vf_f32m1(acc2, *pa, vb6, vl);
        acc7 = vfmacc_vf_f32m1(acc3, *pa, vb7, vl);
        pb_vl += N;
        pa++;
      }

      // update acc with act
      acc0 = vfmin_vf_f32m1(acc0, max, vl);
      acc1 = vfmin_vf_f32m1(acc1, max, vl);
      acc2 = vfmin_vf_f32m1(acc2, max, vl);
      acc3 = vfmin_vf_f32m1(acc3, max, vl);
      acc4 = vfmin_vf_f32m1(acc4, max, vl);
      acc5 = vfmin_vf_f32m1(acc5, max, vl);
      acc6 = vfmin_vf_f32m1(acc6, max, vl);
      acc7 = vfmin_vf_f32m1(acc7, max, vl);

      acc0 = vfmax_vf_f32m1(acc0, min, vl);
      acc1 = vfmax_vf_f32m1(acc1, min, vl);
      acc2 = vfmax_vf_f32m1(acc2, min, vl);
      acc3 = vfmax_vf_f32m1(acc3, min, vl);
      acc4 = vfmax_vf_f32m1(acc4, min, vl);
      acc5 = vfmax_vf_f32m1(acc5, min, vl);
      acc6 = vfmax_vf_f32m1(acc6, min, vl);
      acc7 = vfmax_vf_f32m1(acc7, min, vl);

      vse32_v_f32m1(pc + vl * 0, acc0, vl);
      vse32_v_f32m1(pc + vl * 1, acc1, vl);
      vse32_v_f32m1(pc + vl * 2, acc2, vl);
      vse32_v_f32m1(pc + vl * 3, acc3, vl);
      vse32_v_f32m1(pc + vl * 0, acc4, vl);
      vse32_v_f32m1(pc + vl * 1, acc5, vl);
      vse32_v_f32m1(pc + vl * 2, acc6, vl);
      vse32_v_f32m1(pc + vl * 3, acc7, vl);

      pb += vl;
      pc += vl;
      pbias += vl;
    }

    vl = 0;
    pb = B + NN * 8;
    pc = ptr_out + NN * 8;
    pbias = Bias + NN * 8;
    for (size_t n = REMAIN; n; n -= vl) {
      vl = vsetvl_e32m1(n);
      const float *pa = ptr_a;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc = vle32_v_f32m1(pbias, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m1(pb_vl, vl);
        acc = vfmacc_vf_f32m1(acc, *pa++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc = vfmax_vf_f32m1(vfmin_vf_f32m1(acc, max, vl), min, vl);

      vse32_v_f32m1(pc, acc, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }

    ptr_a += K;
    ptr_out += N;
  }
}

void method_5(size_t M, size_t K, size_t N, const float *A, const float *B,
              const float *Bias, float *C, float min, float max) {
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  size_t MM = M / 2;
  size_t REAMIN = M - MM * 2;
  for (size_t mm = 0; mm < MM; mm++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t n = N; n; n -= vl) {
      vl = vsetvl_e32m8(n);
      const float *pa0 = ptr_a;
      const float *pa1 = pa0 + K;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc0 = vle32_v_f32m8(pbias, vl);
      auto acc1 = acc0;

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m8(pb_vl, vl);
        acc0 = vfmacc_vf_f32m8(acc0, *pa0++, vb, vl);
        acc1 = vfmacc_vf_f32m8(acc1, *pa1++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc0 = vfmax_vf_f32m8(vfmin_vf_f32m8(acc0, max, vl), min, vl);
      acc1 = vfmax_vf_f32m8(vfmin_vf_f32m8(acc1, max, vl), min, vl);

      vse32_v_f32m8(pc + K * 0, acc0, vl);
      vse32_v_f32m8(pc + K * 1, acc1, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }
    ptr_a += K * 2;
    ptr_out += N;
  }
}

float randfloat() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

std::unique_ptr<float[]> randmat(size_t M, size_t N) {
  auto mat = std::make_unique<float[]>(M * N);
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      mat[m * N + n] = randfloat();
    }
  }
  return mat;
}

struct test_method {
  method_t method;
  const char *name;
} methods[] = {
    {method_ref, "ref"}, {method_1, "m1"}, {method_2, "m2"}, {method_3, "m3"}, {method_4, "m4"}, {method_5, "m5"}};

void test(size_t scale, size_t M, size_t K, size_t N) {

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  auto A = randmat(M, K);
  auto B = randmat(K, N);
  auto Bias = randmat(1, N);
  auto C = std::make_unique<float[]>(M * N);
  auto RefC = std::make_unique<float[]>(M * N);
  auto min = randfloat();
  auto max = randfloat();
  auto MAC = M * N * K;

  // ref
  methods[0].method(M, K, N, A.get(), B.get(), Bias.get(), RefC.get(), min,
                    max);

  std::cout << "\nTESTING M=" << M << " K=" << K << " N=" << N << " MAC=" << MAC
            << std::endl;

  for (size_t i = 0; i < sizeof(methods) / sizeof(test_method); i++) {
    auto &m = methods[i];
    memset(C.get(), M * N * sizeof(float), 0);
    // warm up
    for (size_t j = 0; j < WARM_UP_COUNT * scale; j++) {
      m.method(M, K, N, A.get(), B.get(), Bias.get(), C.get(), min, max);
    }

    // check
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        auto diff = std::abs(C[m * N + n] - RefC[m * N + n]);
        if (diff > std::numeric_limits<float>::epsilon()) {
          std::cerr << "Check failed: " << methods[i].name << " M=" << m
                    << " N=" << n << std::endl;
          throw 1;
        }
      }
    }

    // test
    auto t1 = high_resolution_clock::now();
    auto test_count = TEST_COUNT * scale;
    for (size_t j = 0; j < test_count; j++) {
      m.method(M, K, N, A.get(), B.get(), Bias.get(), C.get(), min, max);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    auto t = ms_double.count() / test_count;
    std::cout << m.name << "\t\t" << t << "ms"
              << "\t\t" << (MAC / t / 1e6) << "GMACs" << std::endl;
  }
}

int main() {
  test(20, 16, 16, 16);
  test(1, 16, 128, 128);
}