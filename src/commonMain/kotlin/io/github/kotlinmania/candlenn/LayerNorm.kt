// port-lint: source layer_norm.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor

public data class LayerNormConfig(
    public val eps: Double = 1e-5,
    /**
     * Whether to remove the mean or not, setting this to `false` makes this layer equivalent to
     * RmsNorm.
     */
    public val removeMean: Boolean = true,
    public val affine: Boolean = true,
) {
    public companion object {
        public fun default(): LayerNormConfig = LayerNormConfig()

        public fun fromEps(eps: Double): LayerNormConfig = LayerNormConfig(
            eps = eps,
            removeMean = true,
            affine = true,
        )
    }
}

/**
 * Layer Normalization.
 *
 * This layer applies Layer Normalization over a mini-batch of inputs as described in [Layer
 * Normalization](https://arxiv.org/abs/1607.06450). The input is expected to have three
 * dimensions: a batch dimension, a length, and a hidden size, the normalization is applied over
 * the last dimension.
 */
public class LayerNorm(
    private val weightVal: Tensor,
    private val biasVal: Tensor?,
    private val removeMeanVal: Boolean,
    private val epsVal: Double,
) : Module {
    public companion object {
        public fun new(weight: Tensor, bias: Tensor, eps: Double): LayerNorm =
            LayerNorm(weight, bias, removeMeanVal = true, epsVal = eps)

        public fun newNoBias(weight: Tensor, eps: Double): LayerNorm =
            LayerNorm(weight, null, removeMeanVal = true, epsVal = eps)

        public fun rmsNorm(weight: Tensor, eps: Double): LayerNorm =
            LayerNorm(weight, null, removeMeanVal = false, epsVal = eps)
    }

    public fun weight(): Tensor = weightVal

    public fun bias(): Tensor? = biasVal

    public fun removeMean(): Boolean = removeMeanVal

    public fun isRemoveMean(): Boolean = removeMeanVal

    public fun eps(): Double = epsVal

    override fun forward(xs: Tensor): Tensor {
        if (xs.isContiguous() && removeMeanVal) {
            if (biasVal != null) {
                return layerNorm(xs, weightVal, biasVal, epsVal.toFloat())
            }
        }
        val xDtype = xs.dtype()
        val internalDtype = when (xDtype) {
            DType.F16, DType.BF16 -> DType.F32
            else -> xDtype
        }
        val hiddenSize = xs.dim(D.Minus1)
        val x = xs.toDtype(internalDtype)
        val xSubMean = if (removeMeanVal) {
            val meanX = (x.sumKeepdim(D.Minus1) / hiddenSize.toDouble())
            x.broadcastSub(meanX)
        } else {
            x
        }
        val normX = (xSubMean.sqr().sumKeepdim(D.Minus1) / hiddenSize.toDouble())
        val xNormed = xSubMean.broadcastDiv((normX + epsVal).sqrt())
        val xWeighted = xNormed.toDtype(xDtype).broadcastMul(weightVal)
        return when (biasVal) {
            null -> xWeighted
            else -> xWeighted.broadcastAdd(biasVal)
        }
    }
}

public fun layerNorm(size: Int, config: LayerNormConfig, vb: VarBuilder): LayerNorm {
    val weightName = listOf("weight", "gamma").firstOrNull { vb.containsTensor(it) } ?: "weight"
    val weight = vb.getWithHints(size, weightName, Init.Const(1.0))
    val bias = if (config.affine) {
        val biasName = listOf("bias", "beta").firstOrNull { vb.containsTensor(it) } ?: "bias"
        vb.getWithHints(size, biasName, Init.Const(0.0))
    } else {
        null
    }
    return LayerNorm(
        weightVal = weight,
        biasVal = bias,
        removeMeanVal = config.removeMean,
        epsVal = config.eps,
    )
}

public fun layerNorm(size: Int, eps: Double, vb: VarBuilder): LayerNorm =
    layerNorm(size, LayerNormConfig.fromEps(eps), vb)

public fun layerNormNoBias(size: Int, eps: Double, vb: VarBuilder): LayerNorm {
    val config = LayerNormConfig(
        eps = eps,
        removeMean = true,
        affine = false,
    )
    return layerNorm(size, config, vb)
}

/**
 * Root Mean Square Layer Normalization.
 *
 * This layer applies Root Mean Square Layer Normalization over a mini-batch of inputs as described in [Root Mean Square Layer
 * Normalization](https://arxiv.org/pdf/1910.07467.pdf). The input is expected to have three
 * dimensions: a batch dimension, a length, and a hidden size, the normalization is applied over
 * the last dimension.
 */
public class RmsNorm(
    private val inner: LayerNorm,
) : Module {
    public companion object {
        public fun new(weight: Tensor, eps: Double): RmsNorm =
            RmsNorm(LayerNorm.rmsNorm(weight, eps))
    }

    public fun intoInner(): LayerNorm = inner

    public fun weight(): Tensor = inner.weight()

    public fun eps(): Double = inner.eps()

    public fun forwardDiff(xs: Tensor): Tensor = inner.forward(xs)

    override fun forward(xs: Tensor): Tensor =
        if (xs.isContiguous()) {
            rmsNorm(xs, inner.weight(), inner.eps().toFloat())
        } else {
            inner.forward(xs)
        }
}

public fun rmsNorm(size: Int, eps: Double, vb: VarBuilder): RmsNorm {
    val config = LayerNormConfig(
        eps = eps,
        removeMean = false,
        affine = false,
    )
    return RmsNorm(layerNorm(size, config, vb))
}
