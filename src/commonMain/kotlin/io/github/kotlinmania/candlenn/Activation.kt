// port-lint: source activation.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.DeserializeSeed
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.EnumAccess
import io.github.kotlinmania.serdecore.de.F64Deserialize
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer
import io.github.kotlinmania.serdecore.ser.serialize
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
    Mish,
    HardSwish,
    Elu,
    LeakyRelu,
    GeluPytorchTanh,
}

private class DoubleSerialize(
    private val value: Double,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = value.serialize(serializer)
}

private class SeedOf<T>(
    private val deserialize: Deserialize<T>,
) : DeserializeSeed<T> {
    override fun <D> deserialize(deserializer: D): SerdeResult<T>
        where D : Deserializer = deserialize.deserialize(deserializer)
}

private val ACTIVATION_VARIANTS: List<String> =
    listOf(
        "gelu",
        "newgelu",
        "gelu_new",
        "relu",
        "relu2",
        "relu6",
        "silu",
        "sigmoid",
        "hardsigmoid",
        "swiglu",
        "swish",
        "mish",
        "hardswish",
        "elu",
        "leakyrelu",
        "gelupytorchtanh",
        "gelu_pytorch_tanh",
    )

public class Activation(
    public val kind: ActivationKind = ActivationKind.Gelu,
    public val param: Double = 0.0,
) : Module,
    Serialize {
    public companion object : Deserialize<Activation> {
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
        public val Mish: Activation = Activation(ActivationKind.Mish)
        public val HardSwish: Activation = Activation(ActivationKind.HardSwish)
        public val GeluPytorchTanh: Activation = Activation(ActivationKind.GeluPytorchTanh)

        public fun Elu(alpha: Double): Activation = Activation(ActivationKind.Elu, alpha)

        public fun LeakyRelu(negativeSlope: Double): Activation = Activation(ActivationKind.LeakyRelu, negativeSlope)

        public fun default(): Activation = Gelu

        private fun unitActivationFromName(name: String): Activation? =
            when (name) {
                "gelu" -> Gelu
                "newgelu", "gelu_new" -> NewGelu
                "relu" -> Relu
                "relu2" -> Relu2
                "relu6" -> Relu6
                "silu" -> Silu
                "sigmoid" -> Sigmoid
                "hardsigmoid" -> HardSigmoid
                "swiglu" -> Swiglu
                "swish" -> Swish
                "mish" -> Mish
                "hardswish" -> HardSwish
                "gelupytorchtanh", "gelu_pytorch_tanh" -> GeluPytorchTanh
                else -> null
            }

        override fun <D> deserialize(deserializer: D): SerdeResult<Activation>
            where D : Deserializer =
            deserializer.deserializeEnum(
                "Activation",
                ACTIVATION_VARIANTS,
                object : Visitor<Activation> {
                    override fun expecting(): String = "enum Activation"

                    override fun visitStr(v: String): SerdeResult<Activation> =
                        unitActivationFromName(v)?.let { SerdeResult.success(it) }
                            ?: SerdeResult.failure(SerdeError.unknownVariant(v, ACTIVATION_VARIANTS))

                    override fun visitString(v: String): SerdeResult<Activation> = visitStr(v)

                    override fun <A> visitMap(access: A): SerdeResult<Activation>
                        where A : MapAccess =
                        serdeCatching {
                            val key =
                                access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow()
                                    ?: throw SerdeException(SerdeError.custom("expected variant map"))
                            val result =
                                when (key) {
                                    "elu" -> {
                                        Elu(access.nextValueSeed(SeedOf(F64Deserialize)).getOrThrow())
                                    }

                                    "leakyrelu" -> {
                                        LeakyRelu(access.nextValueSeed(SeedOf(F64Deserialize)).getOrThrow())
                                    }

                                    else -> {
                                        val unitAct =
                                            unitActivationFromName(key)
                                                ?: throw SerdeException(SerdeError.unknownVariant(key, ACTIVATION_VARIANTS))
                                        access.nextValueSeed(SeedOf(io.github.kotlinmania.serdecore.de.UnitDeserialize)).getOrThrow()
                                        unitAct
                                    }
                                }
                            if (access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() != null) {
                                throw SerdeException(SerdeError.custom("unexpected extra key in enum map"))
                            }
                            result
                        }

                    override fun <A> visitEnum(access: A): SerdeResult<Activation>
                        where A : EnumAccess =
                        serdeCatching {
                            val (variantName, variant) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            when (variantName) {
                                "elu" -> {
                                    Elu(variant.newtypeVariant(SeedOf(F64Deserialize)).getOrThrow())
                                }

                                "leakyrelu" -> {
                                    LeakyRelu(variant.newtypeVariant(SeedOf(F64Deserialize)).getOrThrow())
                                }

                                else -> {
                                    val unitAct =
                                        unitActivationFromName(variantName)
                                            ?: throw SerdeException(SerdeError.unknownVariant(variantName, ACTIVATION_VARIANTS))
                                    variant.unitVariant().getOrThrow()
                                    unitAct
                                }
                            }
                        }
                },
            )
    }

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        when (kind) {
            ActivationKind.Gelu -> serializer.serializeUnitVariant("Activation", 0u, "gelu")
            ActivationKind.NewGelu -> serializer.serializeUnitVariant("Activation", 1u, "newgelu")
            ActivationKind.Relu -> serializer.serializeUnitVariant("Activation", 2u, "relu")
            ActivationKind.Relu2 -> serializer.serializeUnitVariant("Activation", 3u, "relu2")
            ActivationKind.Relu6 -> serializer.serializeUnitVariant("Activation", 4u, "relu6")
            ActivationKind.Silu -> serializer.serializeUnitVariant("Activation", 5u, "silu")
            ActivationKind.Sigmoid -> serializer.serializeUnitVariant("Activation", 6u, "sigmoid")
            ActivationKind.HardSigmoid -> serializer.serializeUnitVariant("Activation", 7u, "hardsigmoid")
            ActivationKind.Swiglu -> serializer.serializeUnitVariant("Activation", 8u, "swiglu")
            ActivationKind.Swish -> serializer.serializeUnitVariant("Activation", 9u, "swish")
            ActivationKind.Mish -> serializer.serializeUnitVariant("Activation", 10u, "mish")
            ActivationKind.HardSwish -> serializer.serializeUnitVariant("Activation", 11u, "hardswish")
            ActivationKind.Elu -> serializer.serializeNewtypeVariant("Activation", 12u, "elu", DoubleSerialize(param))
            ActivationKind.LeakyRelu -> serializer.serializeNewtypeVariant("Activation", 13u, "leakyrelu", DoubleSerialize(param))
            ActivationKind.GeluPytorchTanh -> serializer.serializeUnitVariant("Activation", 14u, "gelupytorchtanh")
        }

    override fun forward(xs: Tensor): Tensor =
        when (kind) {
            ActivationKind.Gelu -> {
                xs.geluErf()
            }

            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            // See matching implementation at:
            // https://github.com/ggerganov/llama.cpp/blob/43c9244056fc7c02ab2fe6dcbe2787fd2f2ac3f6/ggml.c#L6098-L6125
            ActivationKind.NewGelu -> {
                xs.gelu()
            }

            ActivationKind.Relu -> {
                xs.relu()
            }

            ActivationKind.Relu2 -> {
                xs.relu().sqr()
            }

            ActivationKind.Relu6 -> {
                xs.clamp(0.0, 6.0)
            }

            ActivationKind.Silu -> {
                xs.silu()
            }

            ActivationKind.Sigmoid -> {
                sigmoid(xs)
            }

            ActivationKind.HardSigmoid -> {
                hardSigmoid(xs)
            }

            ActivationKind.Swiglu -> {
                swiglu(xs)
            }

            ActivationKind.Swish -> {
                xs * sigmoid(xs)
            }

            ActivationKind.HardSwish -> {
                xs * hardSigmoid(xs)
            }

            ActivationKind.Mish -> {
                mish(xs)
            }

            ActivationKind.Elu -> {
                xs.elu(param)
            }

            ActivationKind.LeakyRelu -> {
                leakyRelu(xs, param)
            }

            ActivationKind.GeluPytorchTanh -> {
                val c = sqrt(2.0 / PI)
                0.5 * xs * (1.0 + ((xs + 0.044715 * xs.powf(3.0)) * c).tanh())
            }
        }

    override fun equals(other: Any?): Boolean = other is Activation && kind == other.kind && param == other.param

    override fun hashCode(): Int = 31 * kind.hashCode() + param.hashCode()

    override fun toString(): String = "Activation(kind=$kind, param=$param)"
}

public class PReLU(
    private val weight: Tensor,
    private val isScalar: Boolean,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            isScalar: Boolean,
        ): PReLU = PReLU(weight, isScalar)
    }

    public fun weight(): Tensor = weight

    public fun isScalar(): Boolean = isScalar

    override fun forward(xs: Tensor): Tensor {
        val weight =
            if (isScalar) {
                weight.reshape(Shape(emptyList()))
            } else if (xs.shape() == weight.shape()) {
                weight
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
public fun prelu(
    numChannels: Int?,
    vs: VarBuilder,
): PReLU {
    val initWs = Init.Const(0.25)
    // When using a scalar weight, the PyTorch encoding consists in using a 1d tensor of length 1,
    // we mimic this behavior here so so that checkpoints are compatible.
    val ws = vs.getWithHints(Shape(numChannels ?: 1), "weight", initWs)
    return PReLU.new(ws, numChannels == null)
}
