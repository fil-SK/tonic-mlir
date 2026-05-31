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

#endif // LIB_DIALECT_TONIC_TONICOPS_TD_