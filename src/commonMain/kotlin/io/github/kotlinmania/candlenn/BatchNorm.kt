// port-lint: source batch_norm.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.ModuleT
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import io.github.kotlinmania.candle.bail

public data class BatchNormConfig(
    public val eps: Double = 1e-5,
    public val removeMean: Boolean = true,
    /**
     * The meaning of affine here is different from LayerNorm: when false there is no learnable
     * parameter at all, 1 used for gamma and 0 for beta.
     */
    public val affine: Boolean = true,
    /**
     * Controls exponential moving average of running stats. Defaults to 0.1.
     */
    public val momentum: Double = 0.1,
) {
    public companion object {
        public fun default(): BatchNormConfig = BatchNormConfig()

        public fun from(eps: Double): BatchNormConfig = BatchNormConfig(
            eps = eps,
            removeMean = true,
            affine = true,
            momentum = 0.1,
        )

        public fun fromEps(eps: Double): BatchNormConfig = from(eps)
    }
}

public class BatchNorm internal constructor(
    private val runningMean: Var,
    private val runningVar: Var,
    private val weightAndBias: Pair<Tensor, Tensor>?,
    private val removeMean: Boolean,
    private val eps: Double,
    private val momentum: Double,
) : ModuleT {
    private fun checkValidity(numFeatures: Int) {
        if (eps < 0.0) {
            bail("batch-norm eps cannot be negative $eps")
        }
        if (momentum < 0.0 || momentum > 1.0) {
            bail("batch-norm momentum must be between 0 and 1, is $momentum")
        }
        if (runningMean.asTensor().dims() != listOf(numFeatures)) {
            bail("batch-norm running mean has unexpected shape ${runningMean.asTensor().shape()} should have shape [$numFeatures]")
        }
        if (runningVar.asTensor().dims() != listOf(numFeatures)) {
            bail("batch-norm running variance has unexpected shape ${runningVar.asTensor().shape()} should have shape [$numFeatures]")
        }
        if (weightAndBias != null) {
            val (weight, bias) = weightAndBias
            if (weight.dims() != listOf(numFeatures)) {
                bail("batch-norm weight has unexpected shape ${weight.shape()} should have shape [$numFeatures]")
            }
            if (bias.dims() != listOf(numFeatures)) {
                bail("batch-norm weight has unexpected shape ${bias.shape()} should have shape [$numFeatures]")
            }
        }
    }

    public companion object {
        public fun new(
            numFeatures: Int,
            runningMean: Tensor,
            runningVar: Tensor,
            weight: Tensor,
            bias: Tensor,
            eps: Double,
        ): BatchNorm {
            val out = BatchNorm(
                runningMean = Var.fromTensor(runningMean),
                runningVar = Var.fromTensor(runningVar),
                weightAndBias = Pair(weight, bias),
                removeMean = true,
                eps = eps,
                momentum = 0.1,
            )
            out.checkValidity(numFeatures)
            return out
        }

        public fun newNoBias(
            numFeatures: Int,
            runningMean: Tensor,
            runningVar: Tensor,
            eps: Double,
        ): BatchNorm {
            val out = BatchNorm(
                runningMean = Var.fromTensor(runningMean),
                runningVar = Var.fromTensor(runningVar),
                weightAndBias = null,
                removeMean = true,
                eps = eps,
                momentum = 0.1,
            )
            out.checkValidity(numFeatures)
            return out
        }

        public fun newWithMomentum(
            numFeatures: Int,
            runningMean: Tensor,
            runningVar: Tensor,
            weight: Tensor,
            bias: Tensor,
            eps: Double,
            momentum: Double,
        ): BatchNorm {
            val out = BatchNorm(
                runningMean = Var.fromTensor(runningMean),
                runningVar = Var.fromTensor(runningVar),
                weightAndBias = Pair(weight, bias),
                removeMean = true,
                eps = eps,
                momentum = momentum,
            )
            out.checkValidity(numFeatures)
            return out
        }

        public fun newNoBiasWithMomentum(
            numFeatures: Int,
            runningMean: Tensor,
            runningVar: Tensor,
            eps: Double,
            momentum: Double,
        ): BatchNorm {
            val out = BatchNorm(
                runningMean = Var.fromTensor(runningMean),
                runningVar = Var.fromTensor(runningVar),
                weightAndBias = null,
                removeMean = true,
                eps = eps,
                momentum = momentum,
            )
            out.checkValidity(numFeatures)
            return out
        }
    }

    public fun runningMean(): Tensor = runningMean.asTensor()

    public fun runningVar(): Tensor = runningVar.asTensor()

    public fun eps(): Double = eps

    public fun weightAndBias(): Pair<Tensor, Tensor>? = weightAndBias

    public fun momentum(): Double = momentum

    public fun forwardTrain(x: Tensor): Tensor {
        val numFeatures = runningMean.asTensor().dim(0)
        val xDtype = x.dtype()
        val internalDtype = when (xDtype) {
            DType.F16, DType.BF16 -> DType.F32
            else -> xDtype
        }
        if (x.rank() < 2) {
            bail("batch-norm input tensor must have at least two dimensions (${x.shape()})")
        }
        if (x.dim(1) != numFeatures) {
            bail("batch-norm input doesn't have the expected number of features (${x.shape()} <> $numFeatures)")
        }
        val xCast = x.toDtype(internalDtype)
        val xTransposed = xCast.transpose(0, 1)
        val xDimsPostTranspose = xTransposed.dims()
        // Flatten all the dimensions except the channel one as this performs a Spatial Batch
        // Normalization.
        val xFlat = xTransposed.flattenFrom(1).contiguous()
        val xCentered = if (removeMean) {
            // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
            val meanX = xFlat.meanKeepdim(1)
            val updatedRunningMean = (runningMean.asTensor() * (1.0 - momentum)) +
                (meanX.flattenAll() * momentum)
            runningMean.set(updatedRunningMean)
            xFlat.broadcastSub(meanX)
        } else {
            xFlat
        }
        // The mean is taken over dim 1 as this is the batch dim after the transpose(0, 1) above.
        val normX = xCentered.sqr().meanKeepdim(1)
        val updatedRunningVar = run {
            val batchSize = xCentered.dim(1).toDouble()
            val runningVarWeight = 1.0 - momentum
            val normXWeight = momentum * batchSize / (batchSize - 1.0)
            (runningVar.asTensor() * runningVarWeight) + (normX.flattenAll() * normXWeight)
        }
        runningVar.set(updatedRunningVar)
        val xNormed = xCentered
            .broadcastDiv((normX + eps).sqrt())
            .toDtype(xDtype)
        val xWeighted = when (weightAndBias) {
            null -> xNormed
            else -> {
                val (weight, bias) = weightAndBias
                val weightReshaped = weight.reshape(-1, 1)
                val biasReshaped = bias.reshape(-1, 1)
                xNormed.broadcastMul(weightReshaped).broadcastAdd(biasReshaped)
            }
        }
        return xWeighted.reshape(Shape(xDimsPostTranspose)).transpose(0, 1)
    }

    private fun forwardEval(x: Tensor): Tensor {
        val targetShape = Shape(
            x.dims().mapIndexed { idx, v -> if (idx == 1) v else 1 },
        )
        val xNormed = x
            .broadcastSub(runningMean.asDetachedTensor().reshape(targetShape))
            .broadcastDiv((runningVar.asDetachedTensor().reshape(targetShape) + eps).sqrt())

        return when (weightAndBias) {
            null -> xNormed
            else -> {
                val (weight, bias) = weightAndBias
                val weightReshaped = weight.reshape(targetShape)
                val biasReshaped = bias.reshape(targetShape)
                xNormed.broadcastMul(weightReshaped).broadcastAdd(biasReshaped)
            }
        }
    }

    override fun forwardT(xs: Tensor, train: Boolean): Tensor =
        if (train) {
            forwardTrain(xs)
        } else {
            forwardEval(xs)
        }
}

public fun batchNorm(
    numFeatures: Int,
    config: BatchNormConfig,
    vb: VarBuilder,
): BatchNorm {
    if (config.eps < 0.0) {
        bail("batch-norm eps cannot be negative ${config.eps}")
    }
    val runningMean = vb.getWithHints(numFeatures, "running_mean", Init.Const(0.0))
    val runningVar = vb.getWithHints(numFeatures, "running_var", Init.Const(1.0))
    val weightAndBias = if (config.affine) {
        val weight = vb.getWithHints(numFeatures, "weight", Init.Const(1.0))
        val bias = vb.getWithHints(numFeatures, "bias", Init.Const(0.0))
        Pair(weight, bias)
    } else {
        null
    }
    return BatchNorm(
        runningMean = Var.fromTensor(runningMean),
        runningVar = Var.fromTensor(runningVar),
        weightAndBias = weightAndBias,
        removeMean = config.removeMean,
        eps = config.eps,
        momentum = config.momentum,
    )
}

public fun batchNorm(
    numFeatures: Int,
    eps: Double,
    vb: VarBuilder,
): BatchNorm = batchNorm(numFeatures, BatchNormConfig.from(eps), vb)
