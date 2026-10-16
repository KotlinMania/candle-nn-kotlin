// port-lint: source activation.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.math.PI
import kotlin.math.sqrt

public enum class ActivationKind {
    Gelu,
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    HardSwish,
    Elu,
    LeakyRelu,
    GeluPytorchTanh,
}

public class Activation(
    public val kind: ActivationKind = ActivationKind.Gelu,
    public val param: Double = 0.0,
) : Module {
    public companion object {
        public val Gelu: Activation = Activation(ActivationKind.Gelu)
        public val NewGelu: Activation = Activation(ActivationKind.NewGelu)
        public val Relu: Activation = Activation(ActivationKind.Relu)
        public val Relu2: Activation = Activation(ActivationKind.Relu2)
        public val Relu6: Activation = Activation(ActivationKind.Relu6)
        public val Silu: Activation = Activation(ActivationKind.Silu)
        public val Sigmoid: Activation = Activation(ActivationKind.Sigmoid)
        public val HardSigmoid: Activation = Activation(ActivationKind.HardSigmoid)
        public val Swiglu: Activation = Activation(ActivationKind.Swiglu)
        public val Swish: Activation = Activation(ActivationKind.Swish)
        public val HardSwish: Activation = Activation(ActivationKind.HardSwish)
        public val GeluPytorchTanh: Activation = Activation(ActivationKind.GeluPytorchTanh)

        public fun Elu(alpha: Double): Activation = Activation(ActivationKind.Elu, alpha)
        public fun LeakyRelu(negativeSlope: Double): Activation = Activation(ActivationKind.LeakyRelu, negativeSlope)
        public fun default(): Activation = Gelu
    }

    override fun forward(xs: Tensor): Tensor =
        when (kind) {
            ActivationKind.Gelu -> xs.geluErf()
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            // See matching implementation at:
            // https://github.com/ggerganov/llama.cpp/blob/43c9244056fc7c02ab2fe6dcbe2787fd2f2ac3f6/ggml.c#L6098-L6125
            ActivationKind.NewGelu -> xs.gelu()
            ActivationKind.Relu -> xs.relu()
            ActivationKind.Relu2 -> xs.relu().sqr()
            ActivationKind.Relu6 -> xs.clamp(0.0, 6.0)
            ActivationKind.Silu -> xs.silu()
            ActivationKind.Sigmoid -> sigmoid(xs)
            ActivationKind.HardSigmoid -> hardSigmoid(xs)
            ActivationKind.Swiglu -> swiglu(xs)
            ActivationKind.Swish -> xs * sigmoid(xs)
            ActivationKind.HardSwish -> xs * hardSigmoid(xs)
            ActivationKind.Elu -> xs.elu(param)
            ActivationKind.LeakyRelu -> leakyRelu(xs, param)
            ActivationKind.GeluPytorchTanh -> {
                val c = sqrt(2.0 / PI)
                0.5 * xs * (1.0 + ((xs + 0.044715 * xs.powf(3.0)) * c).tanh())
            }
        }

    override fun equals(other: Any?): Boolean =
        other is Activation && kind == other.kind && param == other.param

    override fun hashCode(): Int = 31 * kind.hashCode() + param.hashCode()

    override fun toString(): String = "Activation(kind=$kind, param=$param)"
}

public class PReLU(
    private val weight: Tensor,
    private val isScalar: Boolean,
) : Module {
    public companion object {
        public fun new(weight: Tensor, isScalar: Boolean): PReLU =
            PReLU(weight, isScalar)
    }

    public fun weight(): Tensor = weight

    public fun isScalar(): Boolean = isScalar

    override fun forward(xs: Tensor): Tensor {
        val weight = if (isScalar) {
            weight.reshape(Shape(emptyList()))
        } else if (xs.rank() >= 2) {
            val numChannels = xs.dim(1)
            val numWeights = weight.elemCount()
            if (numWeights != numChannels) {
                throw io.github.kotlinmania.candle.CandleException.Msg(
                    "error in prelu: unexpected number of channels for the input, got $numChannels, weight is $numWeights",
                )
            }
            val s = MutableList(xs.rank()) { 1 }
            s[1] = weight.elemCount()
            weight.reshape(Shape(s))
        } else {
            weight
        }
        val zeros = xs.zerosLike()
        return xs.maximum(zeros) + xs.minimum(zeros).broadcastMul(weight)
    }
}

/**
 * Create or initialize a new PReLU layer.
 *
 * This uses some default name `weight` for the weight tensor and initializes values to 0.25 if
 * necessary. When `numChannels` is `null`, a single scalar value is used and shared across all
 * channels.
 */
public fun prelu(numChannels: Int?, vs: VarBuilder): PReLU {
    val initWs = Init.Const(0.25)
    // When using a scalar weight, the PyTorch encoding consists in using a 1d tensor of length 1,
    // we mimic this behavior here so so that checkpoints are compatible.
    val ws = vs.getWithHints(Shape(numChannels ?: 1), "weight", initWs)
    return PReLU.new(ws, numChannels == null)
}
