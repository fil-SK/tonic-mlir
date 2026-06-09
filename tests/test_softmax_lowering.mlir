module {
    memref.global "public" @input : memref<2x4xf32> = dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]>
    memref.global "public" @output : memref<2x4xf32>

    func.func @test_softmax() {
        %input = memref.get_global @input : memref<2x4xf32>
        %output = memref.get_global @output : memref<2x4xf32>

        tonic.softmax %input, %output : memref<2x4xf32>, memref<2x4xf32>

        return
    }
}