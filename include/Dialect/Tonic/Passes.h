#ifndef INCLUDE_DIALECT_TONIC_PASSES_H_
#define INCLUDE_DIALECT_TONIC_PASSES_H_

#include "mlir/Pass/Pass.h"


namespace mlir::tonic {
    std::unique_ptr<mlir::Pass> createTonicToLinalgPass();

    #define GEN_PASS_DECL_TONICTOLINALG
    #define GEN_PASS_REGISTRATION
    #include "Dialect/Tonic/Passes.h.inc"
}

#endif // INCLUDE_DIALECT_TONIC_PASSES_H_