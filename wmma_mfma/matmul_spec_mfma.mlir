module attributes { transform.with_named_sequence } {
  transform.named_sequence @transform_codegen(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_matmul -> @transform_annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }

  transform.named_sequence @match_matmul(%entry: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    %c64 = transform.param.constant 64 : i64 -> !transform.param<i64>
    %matmul = transform.match.structured %entry : (!transform.any_op) -> (!transform.any_op) {
    ^bb0(%struct: !transform.any_op):
      transform.match.operation_name %struct ["linalg.matmul_transpose_b"] : !transform.any_op
      %res = transform.match.structured.result %struct[0] {single} : (!transform.any_op) -> !transform.any_op
      transform.match.operation_name %res ["flow.dispatch.tensor.store"] : !transform.any_op
      %dim = transform.match.structured.dim %struct[0] {parallel} : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi ge %dim, %c64 : !transform.param<i64>
      transform.match.structured.yield %struct : !transform.any_op
    }
    transform.yield %matmul : !transform.any_op
  }

  transform.named_sequence @transform_annotate(%matmul: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %matmul {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transform_config = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main> -> !transform.any_param
    transform.annotate %exports "translation_info" = %transform_config : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.consumed}) {
    // Get matmul op
    // ==========================================
    %matmul = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]}
      in %variant_op : (!transform.any_op) -> !transform.any_op
    %tile_sizes0, %tile_sizes1 = transform.iree.create_tile_sizes %matmul : (!transform.any_op) -> (!transform.any_param, !transform.any_param)

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_matmul, %forall_grid =
    transform.structured.tile_using_forall %matmul tile_sizes *(%tile_sizes0 : !transform.any_param)
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Fuse fill
    // ==========================================
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    transform.structured.fuse_into_containing_op %fill into %forall_grid :
      (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func0 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_cse %func0 : !transform.any_op

    // Tile fill
    // ==========================================
    %fill2 = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %tiled_fill3, %forall3 = transform.structured.tile_using_forall %fill2 tile_sizes *(%tile_sizes1 : !transform.any_param) (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile reduction dim
    // ==========================================
    %tiled_matmul2, %loop = transform.structured.tile_using_for %tiled_matmul [0, 0, 32] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote lhs and rhs
    // ==========================================
    %promoted_matmul, %alloc_a, %alloc_b = transform.iree.promote_operands %tiled_matmul2 [0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile to warps
    // ==========================================
    %tiled_matmul3, %forall2 = transform.structured.tile_using_forall %promoted_matmul tile_sizes *(%tile_sizes1 : !transform.any_param) (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.structured.fuse_into_containing_op %fill2 into %forall2 :
    //  (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_cse %func0 : !transform.any_op

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 8, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Reduce bank conflicts by padding
    // ==========================================
    // transform.iree.gpu_reduce_bank_conflicts %func_8 {padding_size_bits = 128} : (!transform.any_op) -> ()

    // Fold ExtF into vector.contract
    // ==========================================
    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    // %new_contract = transform.iree.fold_ext_into_contraction %contract : (!transform.any_op) -> (!transform.any_op)
    transform.iree.apply_cse %func_8 : !transform.any_op

    // Eliminate redundant barriers
    // ==========================================
    transform.iree.eliminate_gpu_barriers %func_8 : (!transform.any_op) -> !transform.any_op

    // Contract to WMMA using layout
    // ==========================================
    %func_9 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %func_9 { use_mfma = true } : (!transform.any_op) -> (!transform.any_op)
    transform.iree.apply_cse %transformed_func : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.iree.apply_cse %func_10 : !transform.any_op

    // Swizzle shared memory
    // ==========================================
    %func_20 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_20 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
      } : !transform.any_op
    transform.iree.optimize_shared_memory_reads_and_writes %func_20 : (!transform.any_op) -> ()

    // Do multi-buffering (num_buffers = pipeline_depth + 1 for loadStoreStage0 (strategy = 1))
    // For now, pipeline depth = 1
    // ==========================================
    //%func_4 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    //transform.iree.gpu_multi_buffering %func_4 {num_buffers = 2, skip_override_analysis = true} : (!transform.any_op) -> ()
    //transform.apply_patterns to %func_4 {
    //    transform.apply_patterns.memref.fold_memref_alias_ops
    //    transform.apply_patterns.canonicalization
    //  } : !transform.any_op

    // Pack shared memory alloc
    // ==========================================
    // transform.iree.pack_shared_memory_alloc %func_10 : (!transform.any_op) -> ()
    // transform.iree.apply_cse %func_10 : !transform.any_op

    // Do pipelining
    // ==========================================
    %for_op = transform.structured.match ops{["scf.for"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    %pipelined_for_op = transform.iree.gpu_pipelining %for_op {depth = 1, strategy = 1, peel_epilogue} : (!transform.any_op) -> (!transform.any_op)

    // Interleave the mma + loads, insert barriers since no multi-buffering
    // ==========================================
    //%func_21 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    //transform.iree.schedule %func_21 : (!transform.any_op) -> ()

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.erase_hal_descriptor_type_from_memref %func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.prepare_vector_to_amd_mma
    } : !transform.any_op

    transform.yield
  }
}
