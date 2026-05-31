module {
    func.func @test_relu_1d(%input: memref<4xf32>, %output: memref<4xf32>) {
        tonic.relu %input, %output : memref<4xf32>, memref<4xf32>
        return
    }

    func.func @test_relu_2d(%input: memref<2x3xf32>, %output: memref<2x3xf32>) {
        tonic.relu %input, %output : memref<2x3xf32>, memref<2x3xf32>
        return
    }
}