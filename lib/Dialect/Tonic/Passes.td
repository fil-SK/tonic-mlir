#ifndef LIB_DIALECT_TONIC_PASSES_TD_
#define LIB_DIALECT_TONIC_PASSES_TD_

include "mlir/Pass/PassBase.td"


def TonicToLinalg : Pass<"tonic-to-linalg", "::mlir::ModuleOp"> {
    let summary = "Lower ops from Tonic dialect onto linalg.generic";
    let description = [{
        Pass rewrites ops from Tonic dialect into an equivalent linalg.generic representation.
    }];
    
    let dependentDialects = [
        "::mlir::linalg::LinalgDialect",
        "::mlir::arith::ArithDialect"
    ];
}

#endif // LIB_DIALECT_TONIC_PASSES_TD_