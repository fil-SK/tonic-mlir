#ifndef LIB_DIALECT_TONIC_TONICOPS_TD_
#define LIB_DIALECT_TONIC_TONICOPS_TD_

include "TonicDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Base class for all Tonic ops
class Tonic_Op<string mnemonic, list<Trait> traits = []> : Op<Tonic_Dialect, mnemonic, traits>;


// Tonic ops
def Tonic_ReluOp : Tonic_Op<"relu"> {
    let summary = "Elementwise ReLU activation function: output[i] = max(0, input[i])";
    let description = [{
        Applies the ReLU (Rectified Linear Unit) activation function, elementwise.
        For every element, calculates: output[i] = max(0, input[i]).

        Constraints:
        - Input and output must have identical static shapes.
        - Element type must be f32.
        - Broadcasting is not supported.
    }];

    let arguments = (ins
        Arg<AnyStaticShapeMemRef, "", [MemRead]>:$input,
        Arg<AnyStaticShapeMemRef, "", [MemWrite]>:$output
    );

    let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}


def Tonic_FlattenOp : Tonic_Op<"flatten"> {
    let summary = "Flatten multi-dimensional memref into 1D memref by copying elements";
    let description = [{
        Applies the flattening by copying all elements into a 1D output memref, using row-major linearization.

        Constraints:
        - ONNX axis parameter is not supported. The op flattens all dimensions.
        - This is a copy operation and not a zero-cost reshape.
    }];

    let arguments = (ins
        Arg<AnyStaticShapeMemRef, "", [MemRead]>:$input,
        Arg<AnyStaticShapeMemRef, "", [MemWrite]>:$output
    );

    let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}


def Tonic_GemmOp : Tonic_Op<"gemm"> {
    let summary = "Performs general matrix multiplication with added bias: Y = A * B + bias";
    let description = [{
        General matrix multiplication, with set constraints, to represent a simplified ONNX Gemm operation.
        Calculates Y = A * B + bias where:
        - A is first input, of shape [M x K]
        - B is second input, of shape [K x N]
        - bias is an array of scalar values, of shape [N], broadcast over M rows (since output is [M x N])
        - Y is output, of shape [M x N]

        Constraints:
        - Parameters locked: alpha = 1, beta = 1, transA = 0, transB = 0
    }];

    let arguments = (ins
        Arg<AnyStaticShapeMemRef, "", [MemRead]>:$a,
        Arg<AnyStaticShapeMemRef, "", [MemRead]>:$b,
        Arg<AnyStaticShapeMemRef, "", [MemRead]>:$bias,
        Arg<AnyStaticShapeMemRef, "", [MemWrite]>:$output
    );

    let assemblyFormat = "$a `,` $b `,` $bias `,` $output attr-dict `:` type($a) `,` type($b) `,` type($bias) `,` type($output)";
}

#endif // LIB_DIALECT_TONIC_TONICOPS_TD_