// Generated code from MLIR EmitC - Copy/pasted output of MLIR processing
float input_1d[4] = {-2.000000000e+00f, -1.000000000e+00f, 5.000000000e+00f, 1.000000000e+01f};
float output_1d[4];
float input_2d[2][3] = {-3.000000000e+00f, -1.000000000e+00f, 5.000000000e+00f, 1.000000000e+01f, -2.000000000e+00f, 1.500000000e+01f};
float output_2d[2][3];
void test_relu_lowering_1d() {
  size_t v1 = 1;
  size_t v2 = 4;
  size_t v3 = 0;
  float v4 = 0.0e+00f;
  for (size_t i5 = v3; i5 < v2; i5 += v1) {
    float v6 = input_1d[i5];
    bool v7 = v6 > v4;
    bool v8 = v6 == v6;
    bool v9 = v4 == v4;
    bool v10 = v8 && v9;
    bool v11 = v10 && v7;
    float v12 = v11 ? v6 : v4;
    output_1d[i5] = v12;
  }
  return;
}

void test_relu_lowering_2d() {
  size_t v1 = 3;
  size_t v2 = 1;
  size_t v3 = 2;
  size_t v4 = 0;
  float v5 = 0.0e+00f;
  for (size_t i6 = v4; i6 < v3; i6 += v2) {
    for (size_t j7 = v4; j7 < v1; j7 += v2) {
      float v8 = input_2d[i6][j7];
      bool v9 = v8 > v5;
      bool v10 = v8 == v8;
      bool v11 = v5 == v5;
      bool v12 = v10 && v11;
      bool v13 = v12 && v9;
      float v14 = v13 ? v8 : v5;
      output_2d[i6][j7] = v14;
    }
  }
  return;
}


// Main runtime
#include <cstdio>

int main() {
    test_relu_lowering_1d();
    //std::cout << "1D ReLU output:\n";
    printf("1D ReLU output:\n");
    for(int i = 0; i < 4; i++) {
        // std::cout << " " << input_1d[i] << " -> " << output_1d[i] << "\n";
        printf("%.1f -> %.1f\n", input_1d[i], output_1d[i]);
    }

    test_relu_lowering_2d();
    // std::cout << "2D ReLU output:\n";
    printf("2D ReLU output:\n");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            //std::cout << " [" << i << "][" << j << "] " << input_2d[i][j] << " -> " << output_2d[i][j] << "\n";
            printf("[%d][%d] %.1f -> %.1f\n", i, j, input_2d[i][j], output_2d[i][j]);
        }
    }
}