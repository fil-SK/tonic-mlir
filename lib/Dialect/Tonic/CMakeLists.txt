# ---------- Dialect ----------
set(LLVM_TARGET_DEFINITIONS TonicDialect.td)
mlir_tablegen(TonicDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TonicDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRTonicDialectIncGen)


# ---------- Operations ----------
set(LLVM_TARGET_DEFINITIONS TonicOps.td)
mlir_tablegen(TonicOps.h.inc -gen-op-decls)
mlir_tablegen(TonicOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTonicOpsIncGen)


# ---------- Passes ----------
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tonic)
add_public_tablegen_target(MLIRTonicPassIncGen)


# ---------- Library ----------
add_mlir_library(TonicDialect
    TonicDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Tonic

    DEPENDS
    MLIRTonicDialectIncGen
    MLIRTonicOpsIncGen
    MLIRTonicPassIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
)