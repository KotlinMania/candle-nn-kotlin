// port-lint: source group_norm.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

/**
 * Group Normalization.
 *
 * This layer applies Group Normalization over a mini-batch of inputs as described in [Group
 * Normalization](https://arxiv.org/abs/1803.08494). The input is expected to have at least three
 * dimensions where the first dimension is the batch size, the second dimension is the number of
 * channels (which must be divisible by `numGroups`) and the remaining dimensions are the spatial
 * dimensions.
 */
public class GroupNorm(
    private val weight: Tensor,
    private val bias: Tensor,
    private val eps: Double,
    private val numChannels: Int,
    private val numGroups: Int,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor,
            numChannels: Int,
            numGroups: Int,
            eps: Double,
        ): GroupNorm {
            if (numChannels % numGroups != 0) {
                bail("GroupNorm: numGroups ($numGroups) must divide numChannels ($numChannels)")
            }
            return GroupNorm(
                weight = weight,
                bias = bias,
                eps = eps,
                numChannels = numChannels,
                numGroups = numGroups,
            )
        }
    }

    public fun weight(): Tensor = weight

    public fun bias(): Tensor = bias

    public fun eps(): Double = eps

    public fun numChannels(): Int = numChannels

    public fun numGroups(): Int = numGroups

    override fun forward(xs: Tensor): Tensor {
        val xShape = xs.dims()
        if (xShape.size <= 2) {
            bail("input rank for GroupNorm should be at least 3, got ${xShape.size}")
        }
        val bSz = xShape[0]
        val nChannels = xShape[1]
        if (nChannels != numChannels) {
            bail("Unexpected numChannels in GroupNorm ($nChannels <> $numChannels)")
        }
        val hiddenSize = xShape.subList(2, xShape.size).fold(1) { a, b -> a * b } * (nChannels / numGroups)

        val xDtype = xs.dtype()
        val internalDtype =
            when (xDtype) {
                DType.F16, DType.BF16 -> DType.F32
                else -> xDtype
            }
        val xsReshaped =
            xs
                .reshape(Shape(bSz, numGroups, hiddenSize))
                .toDtype(internalDtype)
        val meanX = (xsReshaped.sumKeepdim(2) / hiddenSize.toDouble())
        val xsCentered = xsReshaped.broadcastSub(meanX)
        val normX = (xsCentered.sqr().sumKeepdim(2) / hiddenSize.toDouble())
        val wBShapeDims = MutableList(xShape.size) { 1 }
        wBShapeDims[1] = nChannels
        val wBShape = Shape(wBShapeDims)
        val weight = weight.reshape(wBShape)
        val bias = bias.reshape(wBShape)
        return xsCentered
            .broadcastDiv((normX + eps).sqrt())
            .toDtype(xDtype)
            .reshape(Shape(xShape))
            .broadcastMul(weight)
            .broadcastAdd(bias)
    }
}

public fun groupNorm(
    numGroups: Int,
    numChannels: Int,
    eps: Double,
    vb: VarBuilder,
): GroupNorm {
    val weight = vb.getWithHints(numChannels, "weight", Init.Const(1.0))
    val bias = vb.getWithHints(numChannels, "bias", Init.Const(0.0))
    return GroupNorm.new(weight, bias, numChannels, numGroups, eps)
}
