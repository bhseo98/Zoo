// Fused RMSNorm — PORTABLE standard-linalg microkernel.
//
// The normalization is computed in two passes over the last (specialized) dim:
// a squared-sum reduction, then one fused elementwise pass that applies
// rsqrt(mean + eps) and the learned weight. No intermediate normalized tensor is
// materialized separately from the scale. Pure linalg / tensor / arith / math —
// NO iree_linalg_ext — so it lowers portably through IREE (e.g. CPU) today.
//
// TODO(target NPU): target NPU library/runtime IR이 인계되면 이 .mlir의 방출 타깃을
// target NPU intrinsic으로 교체한다. 교체 이음새는 fused_rmsnorm.py의
// MICROKERNEL_BACKEND("standard" → "npu") + 형제 템플릿 fused_rmsnorm_npu.mlir
// 하나뿐이고, CustomOp.select(인터페이스/shape 계약)는 불변이다.

!elem_type = {{elem_type}}
!x_tensor_type = tensor<?x?x{{d}}x!elem_type>
!w_tensor_type = tensor<{{d}}x!elem_type>
!red_tensor_type = tensor<?x?x!elem_type>

module {

util.func private @npu_fused_rmsnorm_{{d}}_{{elem_type}}(
    %x: !x_tensor_type, %w: !w_tensor_type) -> !x_tensor_type {
  %zero = arith.constant 0.0 : !elem_type
  %dim_f = arith.constant {{d_literal}} : !elem_type
  %eps = arith.constant {{eps_literal}} : !elem_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch = tensor.dim %x, %c0 : !x_tensor_type
  %m = tensor.dim %x, %c1 : !x_tensor_type

  // (1) Squared sum over the normalized dim.
  %sum_empty = tensor.empty(%batch, %m) : !red_tensor_type
  %sum_fill = linalg.fill ins(%zero: !elem_type) outs(%sum_empty: !red_tensor_type) -> !red_tensor_type
  %sumsq = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"] }
      ins(%x : !x_tensor_type)
      outs(%sum_fill : !red_tensor_type) {
  ^bb0(%in: !elem_type, %out: !elem_type):
      %sq = arith.mulf %in, %in : !elem_type
      %acc = arith.addf %sq, %out : !elem_type
      linalg.yield %acc : !elem_type
  } -> !red_tensor_type

  // (2) x * rsqrt(mean + eps) * weight — one fused elementwise pass.
  %out_empty = tensor.empty(%batch, %m) : !x_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>,
          affine_map<(d0, d1, d2) -> (d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"] }
      ins(%x, %sumsq, %w : !x_tensor_type, !red_tensor_type, !w_tensor_type)
      outs(%out_empty : !x_tensor_type) {
  ^bb0(%xv: !elem_type, %s: !elem_type, %wv: !elem_type, %o: !elem_type):
      %mean = arith.divf %s, %dim_f : !elem_type
      %shifted = arith.addf %mean, %eps : !elem_type
      %inv = math.rsqrt %shifted : !elem_type
      %norm = arith.mulf %xv, %inv : !elem_type
      %y = arith.mulf %norm, %wv : !elem_type
      linalg.yield %y : !elem_type
  } -> !x_tensor_type

  util.return %result : !x_tensor_type
}

}
