// port-lint: source moe.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

public fun moeGemm(
    input: Tensor,
    weights: Tensor,
    topkWeights: Tensor?,
    sortedTokenIds: Tensor,
    expertsIds: Tensor,
    topk: Int,
    isPrefill: Boolean,
): Tensor {
    if (topk <= 0 || (isPrefill && input.rank() == 0) || weights.rank() == 0 || sortedTokenIds.rank() == 0 || expertsIds.rank() == 0 ||
        topkWeights?.rank() == 0
    ) {
        bail("invalid moeGemm arguments")
    }
    bail("moeGemm is only implemented for the cuda backend")
}

public fun moeGemmGguf(
    input: Tensor,
    weights: QTensor,
    topkWeights: Tensor?,
    sortedTokenIds: Tensor,
    expertsIds: Tensor,
    topk: Int,
    isPrefill: Boolean,
    dtype: DType,
): Tensor {
    if (topk <= 0 || (isPrefill && input.rank() == 0) || weights.rank() == 0 || sortedTokenIds.rank() == 0 || expertsIds.rank() == 0 ||
        topkWeights?.rank() == 0 ||
        dtype.sizeInBytes() == 0
    ) {
        bail("invalid moeGemmGguf arguments")
    }
    bail("moeGemmGguf is only implemented for the cuda backend")
}
