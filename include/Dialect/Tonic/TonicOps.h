#ifndef INCLUDE_DIALECT_TONIC_TONICOPS_H_
#define INCLUDE_DIALECT_TONIC_TONICOPS_H_

#include "Dialect/Tonic/TonicDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


#define GET_OP_CLASSES
#include "Dialect/Tonic/TonicOps.h.inc"

#endif // INCLUDE_DIALECT_TONIC_TONICOPS_H_