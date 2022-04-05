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
  const size_t UNROLL = 8;
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  size_t MM = M / UNROLL;
  size_t REAMIN = M - MM * UNROLL;
  for (size_t mm = 0; mm < MM; mm++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t n = N; n; n -= vl) {
      vl = vsetvl_e32m2(n);
      const float *pa0 = ptr_a;
      const float *pa1 = pa0 + K;
      const float *pa2 = pa0 + K * 2;
      const float *pa3 = pa0 + K * 3;
      const float *pa4 = pa0 + K * 4;
      const float *pa5 = pa0 + K * 5;
      const float *pa6 = pa0 + K * 6;
      const float *pa7 = pa0 + K * 7;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc0 = vle32_v_f32m2(pbias, vl);
      auto acc1 = acc0;
      auto acc2 = acc0;
      auto acc3 = acc0;
      auto acc4 = acc0;
      auto acc5 = acc0;
      auto acc6 = acc0;
      auto acc7 = acc0;

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m2(pb_vl, vl);
        acc0 = vfmacc_vf_f32m2(acc0, *pa0++, vb, vl);
        acc1 = vfmacc_vf_f32m2(acc1, *pa1++, vb, vl);
        acc2 = vfmacc_vf_f32m2(acc2, *pa2++, vb, vl);
        acc3 = vfmacc_vf_f32m2(acc3, *pa3++, vb, vl);
        acc4 = vfmacc_vf_f32m2(acc4, *pa4++, vb, vl);
        acc5 = vfmacc_vf_f32m2(acc5, *pa5++, vb, vl);
        acc6 = vfmacc_vf_f32m2(acc6, *pa6++, vb, vl);
        acc7 = vfmacc_vf_f32m2(acc7, *pa7++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc0 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc0, max, vl), min, vl);
      acc1 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc1, max, vl), min, vl);
      acc2 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc2, max, vl), min, vl);
      acc3 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc3, max, vl), min, vl);
      acc4 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc4, max, vl), min, vl);
      acc5 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc5, max, vl), min, vl);
      acc6 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc6, max, vl), min, vl);
      acc7 = vfmax_vf_f32m2(vfmin_vf_f32m2(acc7, max, vl), min, vl);

      vse32_v_f32m2(pc + K * 0, acc0, vl);
      vse32_v_f32m2(pc + K * 1, acc1, vl);
      vse32_v_f32m2(pc + K * 2, acc2, vl);
      vse32_v_f32m2(pc + K * 3, acc3, vl);
      vse32_v_f32m2(pc + K * 4, acc4, vl);
      vse32_v_f32m2(pc + K * 5, acc5, vl);
      vse32_v_f32m2(pc + K * 6, acc6, vl);
      vse32_v_f32m2(pc + K * 7, acc7, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }
    ptr_a += K * UNROLL;
    ptr_out += N * UNROLL;
  }
}

void method_small(size_t M, size_t K, size_t N, const float *A, const float *B,
                  const float *Bias, float *C, float min, float max) {
  const size_t UNROLL = 4;
  const float *ptr_a = A;
  float *ptr_out = C;
  size_t vl = 0;
  size_t MM = M / UNROLL;
  size_t REAMIN = M - MM * UNROLL;
  for (size_t mm = 0; mm < MM; mm++) {
    const float *pb = B;
    float *pc = ptr_out;
    const float *pbias = Bias;
    for (size_t n = N; n; n -= vl) {
      vl = vsetvl_e32m4(n);
      const float *pa0 = ptr_a;
      const float *pa1 = pa0 + K;
      const float *pa2 = pa0 + K * 2;
      const float *pa3 = pa0 + K * 3;
      const float *pb_vl = pb;

      // init acc with bias
      auto acc0 = vle32_v_f32m4(pbias, vl);
      auto acc1 = vle32_v_f32m4(pbias, vl);
      auto acc2 = vle32_v_f32m4(pbias, vl);
      auto acc3 = vle32_v_f32m4(pbias, vl);

      for (size_t k = 0; k < K; k++) {
        auto vb = vle32_v_f32m4(pb_vl, vl);
        acc0 = vfmacc_vf_f32m4(acc0, *pa0++, vb, vl);
        acc1 = vfmacc_vf_f32m4(acc1, *pa1++, vb, vl);
        acc2 = vfmacc_vf_f32m4(acc2, *pa2++, vb, vl);
        acc3 = vfmacc_vf_f32m4(acc3, *pa3++, vb, vl);
        pb_vl += N;
      }

      // update acc with act
      acc0 = vfmax_vf_f32m4(vfmin_vf_f32m4(acc0, max, vl), min, vl);
      acc1 = vfmax_vf_f32m4(vfmin_vf_f32m4(acc1, max, vl), min, vl);
      acc2 = vfmax_vf_f32m4(vfmin_vf_f32m4(acc2, max, vl), min, vl);
      acc3 = vfmax_vf_f32m4(vfmin_vf_f32m4(acc3, max, vl), min, vl);

      vse32_v_f32m4(pc + K * 0, acc0, vl);
      vse32_v_f32m4(pc + K * 1, acc1, vl);
      vse32_v_f32m4(pc + K * 2, acc2, vl);
      vse32_v_f32m4(pc + K * 3, acc3, vl);
      pb += vl;
      pc += vl;
      pbias += vl;
    }
    ptr_a += K * UNROLL;
    ptr_out += N * UNROLL;
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
} methods[] = {{method_ref, "ref"},
               {method_1, "m1"},
               {method_2, "m2"},
               {method_3, "m3"},
               {method_small, "small"}};

void print_vl(size_t n) {
  std::cout << "\nVL for K=" << n << std::endl;
  std::cout << "m1\t\t" << vsetvl_e32m1(n) << std::endl;
  std::cout << "m2\t\t" << vsetvl_e32m2(n) << std::endl;
  std::cout << "m4\t\t" << vsetvl_e32m4(n) << std::endl;
  std::cout << "m8\t\t" << vsetvl_e32m8(n) << std::endl;
}

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
  auto max = min + 1.f;
  auto MAC = M * N * K;
  auto MEMORY = (M * K + K * N + M * N + N) * sizeof(float);

  print_vl(K);

  // ref
  methods[0].method(M, K, N, A.get(), B.get(), Bias.get(), RefC.get(), min,
                    max);

  std::cout << "\nTESTING M=" << M << " K=" << K << " N=" << N << " MAC=" << MAC
            << " MEMORY=" << MEMORY << " DENSITY=" << ((float)MAC / MEMORY)
            << std::endl;

  for (size_t i = 0; i < sizeof(methods) / sizeof(test_method); i++) {
    auto &method = methods[i];
    memset(C.get(), 0, M * N * sizeof(float));
    // warm up
    for (size_t j = 0; j < WARM_UP_COUNT * scale; j++) {
      method.method(M, K, N, A.get(), B.get(), Bias.get(), C.get(), min, max);
    }

    // check
    for (size_t m = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        auto diff = std::abs(C[m * N + n] - RefC[m * N + n]);
        if (diff > std::numeric_limits<float>::epsilon()) {
          std::cerr << "Check failed: " << method.name << " M=" << m
                    << " N=" << n << " Diff=" << diff << std::endl;
          goto test_b;
        }
      }
    }

  test_b:
    // test
    auto t1 = high_resolution_clock::now();
    auto test_count = TEST_COUNT * scale;
    for (size_t j = 0; j < test_count; j++) {
      method.method(M, K, N, A.get(), B.get(), Bias.get(), C.get(), min, max);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    auto t = ms_double.count() / test_count;
    std::cout << method.name << "\t\t" << t << "ms"
              << "\t\t" << (MAC / t / 1e6) << "GMACs"
              << "\t\t" << (MEMORY / t / 1e6) << "GB" << std::endl;
  }
}

int main() {
  test(20, 16, 16, 16);
  test(20, 16, 32, 32);
  test(1, 16, 128, 128);
}