#ifndef LIB_DIALECT_TONIC_TONICDIALECT_TD_
#define LIB_DIALECT_TONIC_TONICDIALECT_TD_

include "mlir/IR/DialectBase.td"


def Tonic_Dialect : Dialect {
    let name = "tonic";
    let summary = "Tiny ONNX Inference Compiler Dialect";
    let description = [{
        TONIC is a Tiny ONNX Inference Compiler Dialect, which represents a small subset of neural network inference operations.
        It's an entry point of the TONIC compilation pipeline, which performs the lowering from ONNX format to C language, via the EmitC dialect.
    }];
    let cppNamespace = "::mlir::tonic";
}

#endif // LIB_DIALECT_TONIC_TONICDIALECT_TD_