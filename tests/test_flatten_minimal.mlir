// Minimal FlattenOp example
// Just to verify that the parsing is working correctly
func.func @test_flatten_minimal(%input: memref<2x3xf32>, %output: memref<2x3xf32>) {
    tonic.flatten %input, %output : memref<2x3xf32>, memref<2x3xf32>
    return
}