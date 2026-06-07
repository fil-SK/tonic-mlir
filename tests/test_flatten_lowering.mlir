module {
    memref.global "public" @input_2d : memref<2x3xf32> = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
    memref.global "public" @output_1d_from_2d : memref<6xf32>

    memref.global "public" @input_3d : memref<2x3x4xf32> = dense<[
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ],
        [
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0]
        ]
    ]>
    memref.global "public" @output_1d_from_3d : memref<24xf32>


    func.func @test_flatten_2d() {
        %input = memref.get_global @input_2d : memref<2x3xf32>
        %output = memref.get_global @output_1d_from_2d : memref<6xf32>

        tonic.flatten %input, %output : memref<2x3xf32>, memref<6xf32>

        return
    }


    func.func @test_flatten_3d() {
        %input = memref.get_global @input_3d : memref<2x3x4xf32>
        %output = memref.get_global @output_1d_from_3d : memref<24xf32>

        tonic.flatten %input, %output : memref<2x3x4xf32>, memref<24xf32>

        return
    }
}