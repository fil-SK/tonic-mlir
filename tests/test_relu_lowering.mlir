module {
    // For 1D, returns max(0, element), for each element
    func.func @test_relu_lowering_1d(%input : memref<4xf32>, %output : memref<4xf32>) {
        tonic.relu %input, %output : memref<4xf32>, memref<4xf32>
        return
    }

    // For 2D is the same, but uses identity maps for both dimensions
    func.func @test_relu_lowering_2d(%input : memref<2x3xf32>, %output : memref<2x3xf32>) {
        tonic.relu %input, %output : memref<2x3xf32>, memref<2x3xf32>
        return
    }
}