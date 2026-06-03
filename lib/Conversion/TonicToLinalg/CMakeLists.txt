add_mlir_library(TonicToLinalg
    TonicToLinalg.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Dialect/Tonic

    DEPENDS
    MLIRTonicDialectIncGen
    MLIRTonicOpsIncGen
    MLIRTonicPassIncGen

    LINK_LIBS PUBLIC
    TonicDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRIR
    MLIRTransforms
)