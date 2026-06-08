module {
    memref.global "public" @input_A : memref<2x2xf32> = dense<[[1.0, 2.0], [3.0, 4.0]]>
    memref.global "public" @input_B : memref<2x2xf32> = dense<[[1.0, 0.0], [0.0, 1.0]]>
    memref.global "public" @input_bias : memref<2xf32> = dense<[0.5, 0.5]>
    memref.global "public" @output : memref<2x2xf32>

    
    func.func @test_gemm() {
        %input_A = memref.get_global @input_A : memref<2x2xf32>
        %input_B = memref.get_global @input_B : memref<2x2xf32>
        %input_bias = memref.get_global @input_bias : memref<2xf32>
        
        %output = memref.get_global @output : memref<2x2xf32>

        tonic.gemm %input_A, %input_B, %input_bias, %output : memref<2x2xf32>, memref<2x2xf32>, memref<2xf32>, memref<2x2xf32>

        return
    }
}