#include "Dialect/Tonic/TonicDialect.h"
#include "Dialect/Tonic/TonicOps.h"

#include "Dialect/Tonic/TonicDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/Tonic/TonicOps.cpp.inc"

namespace mlir::tonic {

    void TonicDialect::initialize() {
        addOperations<
            #define GET_OP_LIST
            #include "Dialect/Tonic/TonicOps.cpp.inc"
        >();
    }
}