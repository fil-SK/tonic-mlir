#include "Dialect/Tonic/Passes.h"                   // Must be first. Brings in mlir::Pass

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/AffineExpr.h"

#include "Dialect/Tonic/TonicDialect.h"
#include "Dialect/Tonic/TonicOps.h"


namespace mlir::tonic {

    #define GEN_PASS_DEF_TONICTOLINALG
    #include "Dialect/Tonic/Passes.h.inc"
    
    namespace {

        // Conversion rule to handle ReluOp in Tonic dialect
        // OpConversionPattern<ReluOp> will restrict this pattern to just ReluOp
        // which means that matchAndRewrite will not be called on any OTHER type
        struct ReluOpLowering : public OpConversionPattern<ReluOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(ReluOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

                auto loc = op.getLoc();
                auto inputType = cast<MemRefType>(adaptor.getInput().getType());
                int64_t rank = inputType.getRank(); // Get the dimensions

                // For each operand, which indices map to which loop variables 
                // parallel - each iteration is independent
                auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
                SmallVector<AffineMap> maps = {identityMap, identityMap};
                SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

                // Create linalg.generic op
                auto genericOp = rewriter.create<linalg::GenericOp>(
                    loc,
                    TypeRange{},                        // Result types
                    ValueRange{adaptor.getInput()},     // Inputs
                    ValueRange{adaptor.getOutput()},    // Outputs
                    maps,
                    iteratorTypes
                );

                // Create a block inside region in which the code for each iteration runs
                Block *body = rewriter.createBlock(
                    &genericOp.getRegion(),
                    {},
                    {inputType.getElementType(), inputType.getElementType()},
                    {loc, loc}
                );

                // Put new ops at the start of this block
                rewriter.setInsertionPointToStart(body);
                Value inElem = body->getArgument(0);
                Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));

                Value cond = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, inElem, zero);
                Value result = rewriter.create<arith::SelectOp>(loc, cond, inElem, zero);
                rewriter.create<linalg::YieldOp>(loc, result);

                // Delete original tonic.relu op from IR and return success
                rewriter.eraseOp(op);
                return success();
            }
        };


        // Conversion rule for FlattenOp
        struct FlattenOpLowering : public OpConversionPattern<FlattenOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(FlattenOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

                auto loc = op.getLoc();
                auto inputType = cast<MemRefType>(adaptor.getInput().getType());
                int64_t rank = inputType.getRank();
                auto shape = inputType.getShape();

                // Compute the row-major stride
                SmallVector<int64_t> strides(rank);
                strides[rank - 1] = 1;
                for(int i = rank - 2; i >= 0; i--) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }


                // Build the output linearization map
                MLIRContext *ctx = rewriter.getContext();
                AffineExpr linearExpr = getAffineConstantExpr(0, ctx);

                for(int i = 0; i < rank; i++) {
                    linearExpr = linearExpr + getAffineDimExpr(i, ctx) * strides[i];
                }

                auto outputMap = AffineMap::get(rank, 0, linearExpr, ctx);
                auto identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);


                // Again, for each operand, which indices map to which loop variables 
                SmallVector<AffineMap> maps = {identityMap, outputMap};
                SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

                
                // Create linalg.generic op
                auto genericOp = rewriter.create<linalg::GenericOp>(
                    loc,
                    TypeRange{},
                    ValueRange{adaptor.getInput()},
                    ValueRange{adaptor.getOutput()},
                    maps,
                    iteratorTypes
                );


                // Block where the code for each iteration runs
                Block *body = rewriter.createBlock(
                    &genericOp.getRegion(),
                    {},
                    {inputType.getElementType(), inputType.getElementType()},
                    {loc, loc}
                );


                // Here, just copy the element from the input into the linearized position on the output
                rewriter.setInsertionPointToStart(body);
                rewriter.create<linalg::YieldOp>(loc, body->getArgument(0));

                rewriter.eraseOp(op);
                return success();
            }
        };


        struct TonicToLinalgPass : public impl::TonicToLinalgBase<TonicToLinalgPass> {

            void runOnOperation() override {
                MLIRContext *ctx = &getContext();
                ConversionTarget target(*ctx);

                // Output dialects are legal and all Tonic ops must be converted (are illegal)
                target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, memref::MemRefDialect>();
                target.addIllegalDialect<TonicDialect>();

                RewritePatternSet patterns(ctx);
                patterns.add<ReluOpLowering>(ctx);
                patterns.add<FlattenOpLowering>(ctx);

                // partial - ops outside illegal set are left untouched
                // If any tonic operation is present after ALL patterns run then the pass fails
                auto partialConversionResult = applyPartialConversion(getOperation(), target, std::move(patterns));
                if (failed(partialConversionResult)) {
                    signalPassFailure();
                }
            }
        };
    }


    std::unique_ptr<mlir::Pass> createTonicToLinalgPass() {
        return std::make_unique<TonicToLinalgPass>();
    }
    
}