// port-lint: source conv.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CudnnFwdAlgo
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.conv1dWithAlgo
import io.github.kotlinmania.candle.conv2dWithAlgo
import io.github.kotlinmania.candle.convTranspose1d
import io.github.kotlinmania.candle.convTranspose2d
import kotlin.math.sqrt

public data class Conv1dConfig(
    public val padding: Int = 0,
    public val stride: Int = 1,
    public val dilation: Int = 1,
    public val groups: Int = 1,
    public val cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    public companion object {
        public fun default(): Conv1dConfig = Conv1dConfig()
    }
}

/**
 * A 1D convolution layer.
 */
public class Conv1d(
    private val weight: Tensor,
    private val bias: Tensor?,
    private val config: Conv1dConfig,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor?,
            config: Conv1dConfig,
        ): Conv1d = Conv1d(weight, bias, config)
    }

    public fun config(): Conv1dConfig = config

    public fun weight(): Tensor = weight

    public fun bias(): Tensor? = bias

    override fun forward(xs: Tensor): Tensor {
        // The cudnn battle only works for matching channel/kernel sizes, handle the mismatch case here.
        val x =
            if (xs.dim(2) == 1 && weight.dim(2) == 1 && config.stride == 1 && config.padding == 0 && config.dilation == 1 &&
                config.groups == 1
            ) {
                val (bSize, cIn, _) = xs.dims3()
                val cOut = weight.dim(0)
                xs
                    .reshape(Shape(bSize, cIn))
                    .matmul(weight.reshape(Shape(cOut, cIn)).t())
                    .reshape(Shape(bSize, cOut, 1))
            } else {
                xs.conv1dWithAlgo(
                    weight,
                    config.padding,
                    config.stride,
                    config.dilation,
                    config.groups,
                    config.cudnnFwdAlgo,
                )
            }
        return when (bias) {
            null -> {
                x
            }

            else -> {
                val b = bias.dims1()
                val biasReshaped = bias.reshape(Shape(1, b, 1))
                x.broadcastAdd(biasReshaped)
            }
        }
    }
}

public fun conv1d(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: Conv1dConfig,
    vb: VarBuilder,
): Conv1d {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws =
        vb.getWithHints(
            Shape(outChannels, inChannels / cfg.groups, kernelSize),
            "weight",
            initWs,
        )
    val bound = 1.0 / sqrt((inChannels / cfg.groups * kernelSize).toDouble())
    val initBs = Init.Uniform(-bound, bound)
    val bs = vb.getWithHints(outChannels, "bias", initBs)
    return Conv1d.new(ws, bs, cfg)
}

public fun conv1dNoBias(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: Conv1dConfig,
    vb: VarBuilder,
): Conv1d {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws =
        vb.getWithHints(
            Shape(outChannels, inChannels / cfg.groups, kernelSize),
            "weight",
            initWs,
        )
    return Conv1d.new(ws, null, cfg)
}

public data class ConvTranspose1dConfig(
    public val padding: Int = 0,
    public val outputPadding: Int = 0,
    public val stride: Int = 1,
    public val dilation: Int = 1,
    public val groups: Int = 1,
) {
    public companion object {
        public fun default(): ConvTranspose1dConfig = ConvTranspose1dConfig()
    }
}

/**
 * A 1D transposed convolution layer.
 */
public class ConvTranspose1d(
    private val weight: Tensor,
    private val bias: Tensor?,
    private val config: ConvTranspose1dConfig,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor?,
            config: ConvTranspose1dConfig,
        ): ConvTranspose1d = ConvTranspose1d(weight, bias, config)
    }

    public fun config(): ConvTranspose1dConfig = config

    public fun weight(): Tensor = weight

    public fun bias(): Tensor? = bias

    override fun forward(xs: Tensor): Tensor {
        val x =
            xs.convTranspose1d(
                weight,
                config.padding,
                config.outputPadding,
                config.stride,
                config.dilation,
                config.groups,
            )
        return when (bias) {
            null -> {
                x
            }

            else -> {
                val b = bias.dims1()
                val biasReshaped = bias.reshape(Shape(1, b, 1))
                x.broadcastAdd(biasReshaped)
            }
        }
    }
}

public fun convTranspose1d(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
): ConvTranspose1d {
    val bound = 1.0 / sqrt((outChannels / cfg.groups * kernelSize).toDouble())
    val init = Init.Uniform(-bound, bound)
    val ws =
        vb.getWithHints(
            Shape(inChannels, outChannels / cfg.groups, kernelSize),
            "weight",
            init,
        )
    val bs = vb.getWithHints(outChannels, "bias", init)
    return ConvTranspose1d.new(ws, bs, cfg)
}

public fun convTranspose1dNoBias(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
): ConvTranspose1d {
    val bound = 1.0 / sqrt((outChannels / cfg.groups * kernelSize).toDouble())
    val initWs = Init.Uniform(-bound, bound)
    val ws =
        vb.getWithHints(
            Shape(inChannels, outChannels / cfg.groups, kernelSize),
            "weight",
            initWs,
        )
    return ConvTranspose1d.new(ws, null, cfg)
}

public data class Conv2dConfig(
    public val padding: Int = 0,
    public val stride: Int = 1,
    public val dilation: Int = 1,
    public val groups: Int = 1,
    public val cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    public companion object {
        public fun default(): Conv2dConfig = Conv2dConfig()
    }
}

/**
 * A 2D convolution layer.
 */
public class Conv2d(
    private val weight: Tensor,
    private val bias: Tensor?,
    private val config: Conv2dConfig,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor?,
            config: Conv2dConfig,
        ): Conv2d = Conv2d(weight, bias, config)
    }

    public fun config(): Conv2dConfig = config

    public fun weight(): Tensor = weight

    public fun bias(): Tensor? = bias

    public fun absorbBn(bn: BatchNorm): Conv2d {
        val wb =
            bn.weightAndBias() ?: io.github.kotlinmania.candle
                .bail("batch norm does not have weight and bias")
        val (wBn, bBn) = wb
        val stdVal = wBn.div((bn.runningVar() + bn.eps()).sqrt())
        val newWeight = weight.broadcastMul(stdVal.reshape(Shape(weight.dims4().d0, 1, 1, 1)))
        val newBias =
            when (bias) {
                null -> bBn.sub(stdVal.mul(bn.runningMean()))
                else -> bBn.add(stdVal.mul(bias.sub(bn.runningMean())))
            }
        return Conv2d(
            weight = newWeight,
            bias = newBias,
            config = config,
        )
    }

    override fun forward(xs: Tensor): Tensor {
        val x =
            xs.conv2dWithAlgo(
                weight,
                config.padding,
                config.stride,
                config.dilation,
                config.groups,
                config.cudnnFwdAlgo,
            )
        return when (bias) {
            null -> {
                x
            }

            else -> {
                val b = bias.dims1()
                val biasReshaped = bias.reshape(Shape(1, b, 1, 1))
                x.broadcastAdd(biasReshaped)
            }
        }
    }
}

public fun conv2d(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: Conv2dConfig,
    vb: VarBuilder,
): Conv2d {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws =
        vb.getWithHints(
            Shape(outChannels, inChannels / cfg.groups, kernelSize, kernelSize),
            "weight",
            initWs,
        )
    val bound = 1.0 / sqrt((inChannels / cfg.groups * kernelSize * kernelSize).toDouble())
    val initBs = Init.Uniform(-bound, bound)
    val bs = vb.getWithHints(outChannels, "bias", initBs)
    return Conv2d.new(ws, bs, cfg)
}

public fun conv2dNoBias(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: Conv2dConfig,
    vb: VarBuilder,
): Conv2d {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws =
        vb.getWithHints(
            Shape(outChannels, inChannels / cfg.groups, kernelSize, kernelSize),
            "weight",
            initWs,
        )
    return Conv2d.new(ws, null, cfg)
}

public data class ConvTranspose2dConfig(
    public val padding: Int = 0,
    public val outputPadding: Int = 0,
    public val stride: Int = 1,
    public val dilation: Int = 1,
) {
    public companion object {
        public fun default(): ConvTranspose2dConfig = ConvTranspose2dConfig()
    }
}

/**
 * A 2D transposed convolution layer.
 */
public class ConvTranspose2d(
    private val weight: Tensor,
    private val bias: Tensor?,
    private val config: ConvTranspose2dConfig,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor?,
            config: ConvTranspose2dConfig,
        ): ConvTranspose2d = ConvTranspose2d(weight, bias, config)
    }

    public fun config(): ConvTranspose2dConfig = config

    public fun weight(): Tensor = weight

    public fun bias(): Tensor? = bias

    override fun forward(xs: Tensor): Tensor {
        val x =
            xs.convTranspose2d(
                weight,
                config.padding,
                config.outputPadding,
                config.stride,
                config.dilation,
            )
        return when (bias) {
            null -> {
                x
            }

            else -> {
                val b = bias.dims1()
                val biasReshaped = bias.reshape(Shape(1, b, 1, 1))
                x.broadcastAdd(biasReshaped)
            }
        }
    }
}

public fun convTranspose2d(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: ConvTranspose2dConfig,
    vb: VarBuilder,
): ConvTranspose2d {
    val bound = 1.0 / sqrt((outChannels * kernelSize * kernelSize).toDouble())
    val init = Init.Uniform(-bound, bound)
    val ws =
        vb.getWithHints(
            Shape(inChannels, outChannels, kernelSize, kernelSize),
            "weight",
            init,
        )
    val bs = vb.getWithHints(outChannels, "bias", init)
    return ConvTranspose2d.new(ws, bs, cfg)
}

public fun convTranspose2dNoBias(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    cfg: ConvTranspose2dConfig,
    vb: VarBuilder,
): ConvTranspose2d {
    val bound = 1.0 / sqrt((outChannels * kernelSize * kernelSize).toDouble())
    val initWs = Init.Uniform(-bound, bound)
    val ws =
        vb.getWithHints(
            Shape(inChannels, outChannels, kernelSize, kernelSize),
            "weight",
            initWs,
        )
    return ConvTranspose2d.new(ws, null, cfg)
}
