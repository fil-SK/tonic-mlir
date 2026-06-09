#include "Dialect/Tonic/Passes.h"                   // Must be first. Brings in mlir::Pass

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/AffineExpr.h"

#include "Dialect/Tonic/TonicDialect.h"
#include "Dialect/Tonic/TonicOps.h"

#include <limits>                                   // To use -infinity

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


        // Conversion rule for GemmOp
        struct GemmOpLowering : public OpConversionPattern<GemmOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

                auto loc = op.getLoc();
                Value inputA = adaptor.getA();      // Targets the $a from the TableGen definition
                Value inputB = adaptor.getB();
                Value bias = adaptor.getBias();
                Value output = adaptor.getOutput();


                // First, fill the output with zeros
                // Create ConstantOp representing value 0 and then fill the output variable (basically initialize it)
                Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
                rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{output});


                // Perform the matrix multiplication
                rewriter.create<linalg::MatmulOp>(
                    loc,
                    TypeRange{},
                    ValueRange{inputA, inputB},
                    ValueRange{output}
                );


                // Build maps for bias op (through genericop) to use
                MLIRContext *ctx = rewriter.getContext();
                AffineExpr dim1 = getAffineDimExpr(1, ctx);
                auto biasMap = AffineMap::get(2, 0, dim1, ctx);
                auto outputMap = AffineMap::getMultiDimIdentityMap(2, ctx);


                SmallVector<AffineMap> maps = {biasMap, outputMap};
                SmallVector<utils::IteratorType> iterTypes = {
                    utils::IteratorType::parallel,
                    utils::IteratorType::parallel
                };


                // Add the bias parameter onto the calculated matmul
                auto biasOp = rewriter.create<linalg::GenericOp>(
                    loc,
                    TypeRange{},
                    ValueRange{bias},
                    ValueRange{output},
                    maps,
                    iterTypes
                );


                // Create a body block
                auto outputType = cast<MemRefType>(output.getType());
                auto elemType = outputType.getElementType();
                Block *body = rewriter.createBlock(
                    &biasOp.getRegion(),
                    {},
                    {elemType, elemType},
                    {loc, loc}
                );


                // Move cursor into created block and add ADDFOp and YieldOp into the block
                rewriter.setInsertionPointToStart(body);
                Value biasElement = body->getArgument(0);
                Value outputElement = body->getArgument(1);
                Value sum = rewriter.create<arith::AddFOp>(loc, outputElement, biasElement);
                rewriter.create<linalg::YieldOp>(loc, sum);


                rewriter.eraseOp(op);
                return success();
            }
        };


        // Conversion rule for SoftmaxOp
        struct SoftmaxOpLowering : public OpConversionPattern<SoftmaxOp> {
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

                auto loc = op.getLoc();
                Value input = adaptor.getInput();
                Value output = adaptor.getOutput();

                // Extract shape and prepare helper buffer type
                auto inputType = cast<MemRefType>(input.getType());
                auto elementType = inputType.getElementType();
                auto inputShape = inputType.getShape();
                int64_t numOfRows = inputShape[0];
                auto helperBufferType = MemRefType::get({numOfRows}, elementType);

                // Prepare input and row map
                MLIRContext *ctx = rewriter.getContext();
                AffineExpr dim0 = getAffineDimExpr(0, ctx);
                auto inputMap = AffineMap::getMultiDimIdentityMap(2, ctx);
                auto rowMap = AffineMap::get(2, 0, dim0, ctx);

                SmallVector<AffineMap> maps = {inputMap, rowMap};
                SmallVector<utils::IteratorType> iterTypes = {
                    utils::IteratorType::parallel,
                    utils::IteratorType::reduction
                };


                // ---------- STEP 1: Find the maximum value of each row START ----------
                auto negInf = -std::numeric_limits<float>::infinity();
                auto negInfFloatAttr = rewriter.getF32FloatAttr(negInf);

                // First, fill the buffer (basically initialize it) with negative infinity values
                Value maxBuf = rewriter.create<memref::AllocOp>(loc, helperBufferType);     // Allocate the buffer
                Value negativeInfinity = rewriter.create<arith::ConstantOp>(loc, negInfFloatAttr);
                rewriter.create<linalg::FillOp>(loc, ValueRange{negativeInfinity}, ValueRange{maxBuf});

                // Write the reduction, which is max(...), as a GenericOp
                auto maxOp = rewriter.create<linalg::GenericOp>(
                    loc,
                    TypeRange{},
                    ValueRange{input},
                    ValueRange{maxBuf},
                    maps,
                    iterTypes
                );

                // Create body block
                Block *maxOpBody = rewriter.createBlock(
                    &maxOp.getRegion(),
                    {},
                    {elementType, elementType},
                    {loc, loc}
                );

                // Move cursor into created block
                rewriter.setInsertionPointToStart(maxOpBody);

                Value inElement = maxOpBody->getArgument(0);
                Value currentMax = maxOpBody->getArgument(1);
                Value newMax = rewriter.create<arith::MaximumFOp>(loc, inElement, currentMax);

                rewriter.create<linalg::YieldOp>(loc, newMax);
                // ---------- STEP 1: Find the maximum value of each row END ----------
            }
        }


        struct TonicToLinalgPass : public impl::TonicToLinalgBase<TonicToLinalgPass> {

            void runOnOperation() override {
                MLIRContext *ctx = &getContext();
                ConversionTarget target(*ctx);

                // Output dialects are legal and all Tonic ops must be converted (are illegal)
                target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect, memref::MemRefDialect, math::MathDialect>();
                target.addIllegalDialect<TonicDialect>();

                RewritePatternSet patterns(ctx);
                patterns.add<ReluOpLowering>(ctx);
                patterns.add<FlattenOpLowering>(ctx);
                patterns.add<GemmOpLowering>(ctx);

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