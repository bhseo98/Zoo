// NPU INT8 block-scaled matmul — PORTABLE standard-linalg microkernel.
//
// dequant(qs, d) is FUSED into a grouped batch-matmul: the weight stays int8 in
// memory and dequant happens per tile (no full fp weight materialize). Pure
// linalg / tensor / arith / util — NO iree_linalg_ext — so it lowers portably
// through IREE (e.g. CPU) today.
//
// TODO(target NPU): target NPU library/runtime IR이 인계되면 이 .mlir의 방출 타깃을
// target NPU intrinsic으로 교체한다. 교체 이음새는 block_scaled_q8.py의
// MICROKERNEL_BACKEND("standard" → "npu") + 형제 템플릿 block_scaled_q8_npu.mlir
// 하나뿐이고, CustomOp.select(인터페이스/shape 계약)는 불변이다.

{% set accum_type = "f32" %}
!lowp_type = i8
!a_type = {{a_type}}
!scale_type = {{scale_type}}
!accum_type = {{accum_type}}
!a_tensor_type = tensor<?x?x{{k}}x!a_type>
!aexp_tensor_type = tensor<?x?x{{group0}}x{{bs}}x!a_type>
!qs_tensor_type = tensor<{{n}}x{{group0}}x{{bs}}x!lowp_type>
!d_tensor_type = tensor<{{n}}x{{group0}}x1x!scale_type>
!accum_tensor_type = tensor<?x?x{{n}}x!accum_type>
!c_tensor_type = tensor<?x?x{{n}}x!a_type>
!b_grouped_tensor_type = tensor<{{n}}x{{group0}}x{{bs}}x!a_type>

module {

util.func private @npu_mmt_block_scaled_q8_{{n}}_{{k}}_{{bs}}_{{a_type}}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qs: !qs_tensor_type)
    -> !c_tensor_type {
  %zero = arith.constant 0.0: !accum_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch0 = tensor.dim %a, %c0 : !a_tensor_type
  %m = tensor.dim %a, %c1 : !a_tensor_type

  // (1) Dequantize: i8 -> i32 -> fp -> * per-block scale.
  %b_grouped = tensor.empty() : !b_grouped_tensor_type
  %b_grouped_dequant = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"] }
      ins(%d, %qs : !d_tensor_type, !qs_tensor_type)
      outs(%b_grouped : !b_grouped_tensor_type) {
  ^bb0(%d_element: !scale_type, %q_element: !lowp_type, %out: !a_type):
      %q_element_ext = arith.extsi %q_element : !lowp_type to i32
      %q_element_fp = arith.sitofp %q_element_ext : i32 to !a_type
    {% if scale_type == a_type %}
      %q_element_scaled = arith.mulf %q_element_fp, %d_element : !a_type
    {% else %}
      %d_element_ext = arith.extf %d_element : !scale_type to !a_type
      %q_element_scaled = arith.mulf %q_element_fp, %d_element_ext : !a_type
    {% endif %}
      linalg.yield %q_element_scaled : !a_type
  } -> !b_grouped_tensor_type

  // Expand %a to the blocked reduction structure [B, M, group0, bs].
  %aexp = tensor.expand_shape %a [[0], [1], [2, 3]] output_shape [%batch0,%m,{{group0}},{{bs}}] : !a_tensor_type into !aexp_tensor_type

  // (2) Grouped batch matmul: (group0, block) reduction, f32 accum. Dequant (1)
  //     fuses into this tile loop, so the fp weight is never materialized whole.
  %result_empty = tensor.empty(%batch0, %m) : !accum_tensor_type
  %result_fill = linalg.fill ins(%zero: !accum_type) outs(%result_empty: !accum_tensor_type) -> !accum_tensor_type
  %result = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"] }
      ins(%aexp, %b_grouped_dequant : !aexp_tensor_type,  !b_grouped_tensor_type)
      outs(%result_fill : !accum_tensor_type) {
  ^bb0(%a_element: !a_type, %b_element: !a_type, %out: !accum_type):
      %bmm_mul = arith.mulf %a_element, %b_element : !a_type
    {% if accum_type == a_type %}
      %bmm_accum = arith.addf %bmm_mul, %out : !accum_type
    {% else %}
      %bmm_mul_ext = arith.extf %bmm_mul : !a_type to !accum_type
      %bmm_accum = arith.addf %bmm_mul_ext, %out : !accum_type
    {% endif %}
      linalg.yield %bmm_accum : !accum_type
  } -> !accum_tensor_type

  // Cast accum -> output element type.
  %result_cast_empty = tensor.empty(%batch0, %m) : !c_tensor_type
  %result_cast = linalg.copy
    ins(%result : !accum_tensor_type)
    outs(%result_cast_empty : !c_tensor_type) -> !c_tensor_type
  util.return %result_cast : !c_tensor_type
}

}
