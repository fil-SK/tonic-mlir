#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Dialect/Tonic/TonicDialect.h"
#include "Dialect/Tonic/TonicOps.h"
#include "Dialect/Tonic/Passes.h"

int main(int argc, char **argv){
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    // Tonic dialect and passes are registered here, as we add them
    registry.insert<mlir::tonic::TonicDialect>();
    mlir::tonic::registerTonicPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "TONIC Pass Driver\n", registry)
    );
}