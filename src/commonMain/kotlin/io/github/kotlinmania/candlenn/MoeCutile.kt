// port-lint: source moe/cutile.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

private const val MAX_EXPERTS: Int = 1023

/**
 * Selects how input rows map to flattened routes.
 */
public enum class MoeInputMode {
    /** Input has one row per token and each row is reused for all top-k routes. */
    TokenRows,

    /** Input has one row per flattened token-expert route. */
    RoutedRows,
}

internal data class TileConfig(
    val bm: Int,
    val bn: Int,
    val bk: Int,
    val groupM: Int,
)

internal fun defaultConfig(numTokens: Int, numExperts: Int): TileConfig {
    val bm = when {
        numTokens <= 32 -> 16
        numTokens <= 96 -> 32
        numTokens <= 512 -> 64
        else -> 128
    }
    return TileConfig(
        bm = bm,
        bn = if (numTokens <= 64) 64 else 128,
        bk = if (numTokens <= 64) 128 else 64,
        groupM = if (numTokens / numExperts > 128) 16 else 1,
    )
}

/**
 * CUDA routing metadata shared by multiple grouped matrix multiplications.
 */
public class MoeRouting private constructor(
    private val numTokensVal: Int,
    private val topKVal: Int,
    private val numExpertsVal: Int,
) {
    public fun numTokens(): Int = numTokensVal

    public fun topK(): Int = topKVal

    public fun numExperts(): Int = numExpertsVal

    public companion object {
        public fun new(topkIds: Tensor, numExperts: Int): MoeRouting {
            val (numTokens, topK) = topkIds.dims2()
            if (numTokens == 0 || topK == 0) {
                bail("cuTile MoE routing requires at least one route")
            }
            if (numExperts == 0 || numExperts > MAX_EXPERTS) {
                bail("cuTile MoE supports between 1 and $MAX_EXPERTS experts")
            }
            if (topkIds.dtype() != DType.U32 && topkIds.dtype() != DType.I32) {
                bail("cuTile MoE top-k ids must be u32 or i32")
            }
            bail("cuTile MoE routing requires CUDA backend")
        }
    }
}

public fun routedGroupedMatmul(
    input: Tensor,
    expertWeights: Tensor,
    routing: MoeRouting,
    inputMode: MoeInputMode,
    routeWeights: Tensor? = null,
): Tensor {
    if (input.rank() == 0 || expertWeights.rank() == 0 || routing.numExperts() <= 0 || inputMode.name.isEmpty() || routeWeights?.rank() == 0) {
        bail("invalid routedGroupedMatmul arguments")
    }
    bail("cuTile routed grouped matmul requires CUDA backend")
}

public fun warmupRoutedGroupedMatmul(
    input: Tensor,
    expertWeights: Tensor,
    routing: MoeRouting,
    inputMode: MoeInputMode,
    routeWeights: Tensor? = null,
) {
    if (input.rank() == 0 || expertWeights.rank() == 0 || routing.numExperts() <= 0 || inputMode.name.isEmpty() || routeWeights?.rank() == 0) {
        bail("invalid warmupRoutedGroupedMatmul arguments")
    }
    bail("cuTile routed grouped matmul requires CUDA backend")
}
