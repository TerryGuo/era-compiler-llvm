//===--- SyncVMLoopIndexedLdStRecognize.cpp - Favor indexed ld/st ---------===//
//
// \file
// This pass intends to utilize the SCEV info to find load/store that can be
// optimized to indexed load/store provided by SyncVM.
//
// Please be noted:
//   We don't generated indexed ld/st in current pass. We just re-write the IR
//   to favor the subsequent SyncVMCombineToIndexedMemops pass which will
//   generate indexed ld/st in return.
//============================================================================//

#include "SyncVM.h"

#include "SyncVMSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"

#include "llvm/IR/IRBuilder.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/CodeGen/TargetPassConfig.h"

#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "syncvm-loop-indexed-memops-recognize"
#define SYNCVM_RECOGNIZE_INDEXED_MEMOPS_NAME                                   \
  "SyncVM recognize instructions to generate indexed memory operations"

namespace {

class SyncVMLoopIndexedLdStRecognize : public LoopPass {

  ScalarEvolution *SE = nullptr;
  LLVMContext *Ctx = nullptr;

public:
  static char ID;

  SyncVMLoopIndexedLdStRecognize() : LoopPass(ID) {}

  bool runOnLoop(Loop *L, LPPassManager &) override;

  StringRef getPassName() const override {
    return "SyncVM Recognize Indexed Load/Store";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }

private:
  bool IsIncByOneCell(Value *BasePtrValue);
  bool RewriteToFavorIndexedLdST(Value *BasePtrValue, Instruction *currentI,
                                 BasicBlock *currentBB, Loop *currentL);
};

} // end namespace

// Now we know that the baseptr of load/store instruction will be increased
// by one cell via GEP instruction with loop-index as index operand.
//
// This function intends to re-write this GEP instruction and decouple its
// increasing via loop-index.
// The baseptr will be increased via an explicit way which can meet
// the matching mechanism in the subsequent CombineToIndexedMemops pass,
// the indexed load/store will be generated then.
bool SyncVMLoopIndexedLdStRecognize::RewriteToFavorIndexedLdST(
    Value *BasePtrValue, Instruction *currentI, BasicBlock *currentBB,
    Loop *currentL) {
  IRBuilder<> Builder(currentBB);
  Type *TypeOfCopyLen = IntegerType::getInt256Ty(*Ctx);

  auto *GEPInst = dyn_cast<GetElementPtrInst>(BasePtrValue);
  Value *SrcOperand = getPointerOperand(GEPInst);
  Type *SrcType = GEPInst->getPointerOperandType();

  // Generate a new BasePtr.
  Builder.SetInsertPoint(currentBB->getFirstNonPHI());
  auto *NewBasePtrValue = Builder.CreatePHI(SrcType, 2, "baseptr");

  if (isa<Instruction>(SrcOperand)) {
    NewBasePtrValue->addIncoming(SrcOperand, currentL->getLoopPreheader());
  } else {
    Builder.SetInsertPoint(currentL->getLoopPreheader()->getTerminator());
    Value *NewGEPInitInst =
        Builder.CreateInBoundsGEP(GEPInst->getResultElementType(), SrcOperand,
                                  ConstantInt::get(TypeOfCopyLen, 0U));
    NewBasePtrValue->addIncoming(NewGEPInitInst, currentL->getLoopPreheader());
  }

  // Replace the current BasePtr with this new BasePtr.
  GEPInst->replaceAllUsesWith(NewBasePtrValue);
  GEPInst->eraseFromParent();

  // Make a new instruction to increase NewBasePtr by one cell
  // in a way without using loop-index.
  Builder.SetInsertPoint(currentI);
  auto *IncNewBasePtrValue = Builder.CreateInBoundsGEP(
      SrcType, NewBasePtrValue, ConstantInt::get(TypeOfCopyLen, 1U));

  // Add to the PHI
  NewBasePtrValue->addIncoming(IncNewBasePtrValue, currentBB);

  return true;
}

bool SyncVMLoopIndexedLdStRecognize::IsIncByOneCell(Value *BasePtrValue) {
  const SCEV *Ptr = SE->getSCEV(BasePtrValue);

  if (!isa<GetElementPtrInst>(BasePtrValue))
    return false;

  if (Ptr) {
    const SCEVAddRecExpr *AddRec = cast<SCEVAddRecExpr>(Ptr);
    const SCEVConstant *Step =
        dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(*SE));
    if (!Step)
      return false;
    APInt StrideVal = Step->getAPInt();
    if (StrideVal != 32)
      return false;
  }

  return true;
}

bool SyncVMLoopIndexedLdStRecognize::runOnLoop(Loop *L, LPPassManager &) {
  bool Changed = false;

  if (skipLoop(L))
    return false;

  if (!L->isLoopSimplifyForm())
    return false;

  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  Ctx = &L->getLoopPreheader()->getContext();

  for (const auto BB : L->blocks()) {
    for (auto &I : *BB) {
      Value *BasePtrValue = nullptr;
      if (LoadInst *LMemI = dyn_cast<LoadInst>(&I)) {
        BasePtrValue = LMemI->getPointerOperand();
      } else if (StoreInst *SMemI = dyn_cast<StoreInst>(&I)) {
        BasePtrValue = SMemI->getPointerOperand();
      } else {
        continue; // Skip if current inst isn't load nor store inst.
      }

      // Use SCEV info to check whether baseptr is increased by one cell
      if (!IsIncByOneCell(BasePtrValue))
        continue;

      // Let's try to rewrite the GEP instruction in a way that will
      // favor the subsequent CombineToIndexedMemops pass.
      Changed |= RewriteToFavorIndexedLdST(BasePtrValue, &I, BB, L);
    }
  }

  return Changed;
}

Pass *llvm::createSyncVMLoopIndexedLdStRecognizePass() {
  return new SyncVMLoopIndexedLdStRecognize();
}

char SyncVMLoopIndexedLdStRecognize::ID = 0;

INITIALIZE_PASS_BEGIN(SyncVMLoopIndexedLdStRecognize, DEBUG_TYPE,
                      SYNCVM_RECOGNIZE_INDEXED_MEMOPS_NAME, false, false)
INITIALIZE_PASS_END(SyncVMLoopIndexedLdStRecognize, DEBUG_TYPE,
                    SYNCVM_RECOGNIZE_INDEXED_MEMOPS_NAME, false, false)
