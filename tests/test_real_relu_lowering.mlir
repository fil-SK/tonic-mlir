module {
    memref.global "public" @input_1d : memref<4xf32> = dense<[-2.0, -1.0, 5.0, 10.0]>
    memref.global "public" @output_1d : memref<4xf32>

    memref.global "public" @input_2d : memref<2x3xf32> = dense<[[-3.0, -1.0, 5.0], [10.0, -2.0, 15.0]]>
    memref.global "public" @output_2d : memref<2x3xf32>


    // 1D test
    func.func @test_relu_lowering_1d() {
        %input = memref.get_global @input_1d : memref<4xf32>
        %output = memref.get_global @output_1d : memref<4xf32>
        
        tonic.relu %input, %output : memref<4xf32>, memref<4xf32>

        return
    }

    // 2D test
    func.func @test_relu_lowering_2d() {
        %input = memref.get_global @input_2d : memref<2x3xf32>
        %output = memref.get_global @output_2d : memref<2x3xf32>
        
        tonic.relu %input, %output : memref<2x3xf32>, memref<2x3xf32>

        return
    }
}