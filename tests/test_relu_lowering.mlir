module {
    // For 1D, returns max(0, element), for each element
    // Input and output are stack-allocated inside the function
    // Values cover both negative (clamp to 0) and positive (returns those) cases
    func.func @test_relu_lowering_1d() {
        %input = memref.alloca() : memref<4xf32>
        %output = memref.alloca() : memref<4xf32>

        // Indices
        %constant_0 = arith.constant 0 : index
        %constant_1 = arith.constant 1 : index
        %constant_2 = arith.constant 2 : index
        %constant_3 = arith.constant 3 : index

        // Values
        %negative_1 = arith.constant -1.0 : f32
        %negative_2 = arith.constant -2.0 : f32

        %positive_5 = arith.constant 5.0 : f32
        %positive_10 = arith.constant 10.0 : f32

        // Store values
        memref.store %negative_2, %input[%constant_0] : memref<4xf32>
        memref.store %negative_1, %input[%constant_1] : memref<4xf32>
        memref.store %positive_5, %input[%constant_2] : memref<4xf32>
        memref.store %positive_10, %input[%constant_3] : memref<4xf32>

        tonic.relu %input, %output : memref<4xf32>, memref<4xf32>

        return
    }

    // For 2D is the same, but uses identity maps for both dimensions
    func.func @test_relu_lowering_2d() {
        %input = memref.alloca() : memref<2x3xf32>
        %output = memref.alloca() : memref<2x3xf32>

        // Indices
        %constant_0 = arith.constant 0 : index
        %constant_1 = arith.constant 1 : index
        %constant_2 = arith.constant 2 : index

        // Values
        %negative_1 = arith.constant -1.0 : f32
        %negative_2 = arith.constant -2.0 : f32
        %negative_3 = arith.constant -3.0 : f32

        %positive_5 = arith.constant 5.0 : f32
        %positive_10 = arith.constant 10.0 : f32
        %positive_15 = arith.constant 15.0 : f32

        // Store values
        memref.store %negative_3, %input[%constant_0, %constant_0] : memref<2x3xf32>
        memref.store %negative_1, %input[%constant_0, %constant_1] : memref<2x3xf32>
        memref.store %positive_5, %input[%constant_0, %constant_2] : memref<2x3xf32>
        memref.store %positive_10, %input[%constant_1, %constant_0] : memref<2x3xf32>
        memref.store %negative_2, %input[%constant_1, %constant_1] : memref<2x3xf32>
        memref.store %positive_15, %input[%constant_1, %constant_2] : memref<2x3xf32>
        
        tonic.relu %input, %output : memref<2x3xf32>, memref<2x3xf32>

        return
    }
}