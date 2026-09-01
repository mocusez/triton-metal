// RUN: triton-metal-opt --convert-tritongpu-to-metal %s | FileCheck %s
//
// Mixed pointer alignment in one elementwise kernel: aligned x/y
// (tt.divisibility = 16 -> sizePerThread = [4]) and an unaligned out
// (sizePerThread = [1], e.g. an MPS tensor sliced to base[7:]). The Triton
// frontend bridges the spt=4 compute value to the spt=1 store with
// `ttg.convert_layout #blocked -> #blocked1`. Before normalizeRank1DivergentCvts
// this hit the L1d3 "broader staged-transpose deferred" hard error; now the
// cvt's producer cone (loads, addf, addptr/splat/make_range, mask cmpi) is
// rewritten to the spt=1 encoding so loads and store share strided indexing and
// the cvt collapses to an identity (passthrough). See driver.py / the MPS
// storage-offset test for the runtime motivation.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_mixed_align(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32>, %n: i32 {tt.divisibility = 16 : i32}) {
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %off = arith.muli %pid, %c1024_i32 : i32
    %r0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %r1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %s0 = tt.splat %off : i32 -> tensor<1024xi32, #blocked>
    %s1 = tt.splat %off : i32 -> tensor<1024xi32, #blocked1>
    %o0 = arith.addi %s0, %r0 : tensor<1024xi32, #blocked>
    %o1 = arith.addi %s1, %r1 : tensor<1024xi32, #blocked1>
    %ns = tt.splat %n : i32 -> tensor<1024xi32, #blocked>
    %ns1 = tt.splat %n : i32 -> tensor<1024xi32, #blocked1>
    %m0 = arith.cmpi slt, %o0, %ns : tensor<1024xi32, #blocked>
    %m1 = arith.cmpi slt, %o1, %ns1 : tensor<1024xi32, #blocked1>
    %xs = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %xp = tt.addptr %xs, %o0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %xv = tt.load %xp, %m0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %ys = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %yp = tt.addptr %ys, %o0 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %yv = tt.load %yp, %m0 : tensor<1024x!tt.ptr<f32>, #blocked>
    %os = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %op = tt.addptr %os, %o1 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    %sum = arith.addf %xv, %yv : tensor<1024xf32, #blocked>
    %cvt = ttg.convert_layout %sum : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    tt.store %op, %cvt, %m1 : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }

  // Exact producer-sharing shape emitted by medium-stream-compaction.py for
  // aligned MPS tensors. The loaded value and predicate feed both the spt=1
  // scan/scatter path and source-layout users, so the destination-layout
  // branches must be cloned and genuinely re-encoded instead of forwarded as
  // scalar identities.
  tt.func public @stream_compaction_shared_cone(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %block_offsets_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
    %c1024 = arith.constant 1024 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c1024 : i32
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %start = tt.splat %block_start : i32 -> tensor<1024xi32, #blocked>
    %offsets = arith.addi %start, %range : tensor<1024xi32, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<1024xi32, #blocked>
    %in_bounds = arith.cmpi slt, %offsets, %n_splat : tensor<1024xi32, #blocked>
    %a_base = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %a_addresses = tt.addptr %a_base, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %a = tt.load %a_addresses, %in_bounds, %zero : tensor<1024x!tt.ptr<f32>, #blocked>
    %valid = arith.cmpf ogt, %a, %zero : tensor<1024xf32, #blocked>
    %valid_scan = ttg.convert_layout %valid : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    %valid_i32 = arith.extui %valid_scan : tensor<1024xi1, #blocked1> to tensor<1024xi32, #blocked1>
    %inclusive = "tt.scan"(%valid_i32) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.scan.return %sum : i32
    }) : (tensor<1024xi32, #blocked1>) -> tensor<1024xi32, #blocked1>
    %local_offsets = arith.subi %inclusive, %valid_i32 : tensor<1024xi32, #blocked1>
    %block_offset_ptr = tt.addptr %block_offsets_ptr, %pid : !tt.ptr<i32>, i32
    %block_offset = tt.load %block_offset_ptr : !tt.ptr<i32>
    %block_offset_splat = tt.splat %block_offset : i32 -> tensor<1024xi32, #blocked1>
    %write_offsets = arith.addi %block_offset_splat, %local_offsets : tensor<1024xi32, #blocked1>
    %write_mask = arith.andi %valid, %in_bounds : tensor<1024xi1, #blocked>
    %out_base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %out_addresses = tt.addptr %out_base, %write_offsets : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    %store_value = ttg.convert_layout %a : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    %store_mask = ttg.convert_layout %write_mask : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    tt.store %out_addresses, %store_value, %store_mask : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }

  // Radix-style scatter: unlike stream compaction, both pointer and value are
  // relabeled. The masked-store lowering peels this pair and stores in the
  // source layout. The generic normalizer must not independently re-encode the
  // pointer cone through its scan boundary.
  tt.func public @scan_scatter_paired_relabels(%src_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
    %zero = arith.constant dense<0> : tensor<1024xi32, #blocked>
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %range1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %n_splat = tt.splat %n : i32 -> tensor<1024xi32, #blocked>
    %n_splat1 = tt.splat %n : i32 -> tensor<1024xi32, #blocked1>
    %valid = arith.cmpi slt, %range, %n_splat : tensor<1024xi32, #blocked>
    %valid1 = arith.cmpi slt, %range1, %n_splat1 : tensor<1024xi32, #blocked1>
    %src_base = tt.splat %src_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %src_addresses = tt.addptr %src_base, %range : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %values = tt.load %src_addresses, %valid, %zero : tensor<1024x!tt.ptr<i32>, #blocked>
    %selected = arith.cmpi sgt, %values, %zero : tensor<1024xi32, #blocked>
    %selected_i32 = arith.extui %selected : tensor<1024xi1, #blocked> to tensor<1024xi32, #blocked>
    %inclusive = "tt.scan"(%selected_i32) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.scan.return %sum : i32
    }) : (tensor<1024xi32, #blocked>) -> tensor<1024xi32, #blocked>
    %write_offsets = arith.subi %inclusive, %selected_i32 : tensor<1024xi32, #blocked>
    %dst_base = tt.splat %dst_ptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>, #blocked>
    %dst_addresses = tt.addptr %dst_base, %write_offsets : tensor<1024x!tt.ptr<i32>, #blocked>, tensor<1024xi32, #blocked>
    %store_addresses = ttg.convert_layout %dst_addresses : tensor<1024x!tt.ptr<i32>, #blocked> -> tensor<1024x!tt.ptr<i32>, #blocked1>
    %store_values = ttg.convert_layout %values : tensor<1024xi32, #blocked> -> tensor<1024xi32, #blocked1>
    tt.store %store_addresses, %store_values, %valid1 : tensor<1024x!tt.ptr<i32>, #blocked1>
    tt.return
  }

  // The raw reverse workload has two stores of values loaded in #blocked.
  // Triton gives the second store a #blocked1 address/mask and bridges only
  // its value.  Rewriting the shared producer cone made the first store use
  // strided addresses with contiguous reverse-loaded values.  Store-side
  // normalization instead moves the second address/mask to #blocked, so every
  // load and store uses one contiguous ownership mapping.
  tt.func public @reverse_shared_index_cones(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32) {
    %c1024 = arith.constant 1024 : i32
    %minus_one = arith.constant -1 : i32
    %two = arith.constant 2 : i32
    %zero = arith.constant dense<0> : tensor<1024xi32, #blocked>
    %zero1 = arith.constant dense<0> : tensor<1024xi32, #blocked1>
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %c1024 : i32
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %range1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
    %base_splat = tt.splat %base : i32 -> tensor<1024xi32, #blocked>
    %base_splat1 = tt.splat %base : i32 -> tensor<1024xi32, #blocked1>
    %offsets = arith.addi %base_splat, %range : tensor<1024xi32, #blocked>
    %offsets1 = arith.addi %base_splat1, %range1 : tensor<1024xi32, #blocked1>
    %x_splat = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %left_ptr = tt.addptr %x_splat, %offsets : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %end_ptr = tt.addptr %x_ptr, %n : !tt.ptr<f32>, i32
    %last_ptr = tt.addptr %end_ptr, %minus_one : !tt.ptr<f32>, i32
    %last_splat = tt.splat %last_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %last_splat1 = tt.splat %last_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked1>
    %negative = arith.subi %zero1, %offsets1 : tensor<1024xi32, #blocked1>
    %right_ptr1 = tt.addptr %last_splat1, %negative : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
    %half = arith.divsi %n, %two : i32
    %half_splat = tt.splat %half : i32 -> tensor<1024xi32, #blocked>
    %half_splat1 = tt.splat %half : i32 -> tensor<1024xi32, #blocked1>
    %mask = arith.cmpi slt, %offsets, %half_splat : tensor<1024xi32, #blocked>
    %mask1 = arith.cmpi slt, %offsets1, %half_splat1 : tensor<1024xi32, #blocked1>
    %left = tt.load %left_ptr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %negative_src = arith.subi %zero, %offsets : tensor<1024xi32, #blocked>
    %right_ptr = tt.addptr %last_splat, %negative_src : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %right = tt.load %right_ptr, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %left_ptr, %right, %mask : tensor<1024x!tt.ptr<f32>, #blocked>
    %left1 = ttg.convert_layout %left : tensor<1024xf32, #blocked> -> tensor<1024xf32, #blocked1>
    tt.store %right_ptr1, %left1, %mask1 : tensor<1024x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// The pass must succeed (previously a hard error) and emit a Metal kernel; no
// ttg.convert_layout survives the normalization + passthrough.
// CHECK-LABEL: metal.kernel add_mixed_align
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.return

// CHECK-LABEL: metal.kernel stream_compaction_shared_cone
// CHECK: metal.threadgroup_prefix_sum
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.return

// CHECK-LABEL: metal.kernel scan_scatter_paired_relabels
// CHECK: metal.threadgroup_prefix_sum
// CHECK-NOT: ttg.convert_layout
// CHECK: metal.return

// CHECK-LABEL: metal.kernel reverse_shared_index_cones
// CHECK-NOT: ttg.convert_layout
// CHECK: scf.for %[[IV:[^ ]+]] =
// CHECK-NOT: arith.muli %[[IV]],
// CHECK: metal.return
