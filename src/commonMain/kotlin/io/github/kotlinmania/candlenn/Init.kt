// port-lint: source init.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import kotlin.math.sqrt

/**
 * Number of features as input or output of a layer.
 * In Kaiming initialization, choosing [FanIn] preserves
 * the magnitude of the variance of the weights in the
 * forward pass, choosing [FanOut] preserves this
 * magnitude in the backward pass.
 */
public enum class FanInOut {
    FanIn,
    FanOut,
    ;

    /**
     * Compute the fan-in or fan-out value for a weight tensor of
     * the specified dimensions.
     */
    public fun forShape(shape: Shape): Int {
        val dims = shape.dims()
        val receptiveFieldSize = dims.drop(2).fold(1) { acc, d -> acc * d }
        return when (this) {
            FanIn -> {
                if (dims.size < 2) {
                    1
                } else {
                    dims[1] * receptiveFieldSize
                }
            }
            FanOut -> {
                if (dims.isEmpty()) {
                    1
                } else {
                    dims[0] * receptiveFieldSize
                }
            }
        }
    }
}

public enum class NormalOrUniform {
    Normal,
    Uniform,
}

public enum class NonLinearityKind {
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain,
}

/**
 * The non-linear function that follows this layer. ReLU is the
 * recommended value.
 */
public class NonLinearity(
    public val kind: NonLinearityKind,
    public val explicitGainValue: Double = 1.0,
) {
    public fun gain(): Double =
        when (kind) {
            NonLinearityKind.ReLU -> sqrt(2.0)
            NonLinearityKind.Tanh -> 5.0 / 3.0
            NonLinearityKind.Linear, NonLinearityKind.Sigmoid -> 1.0
            NonLinearityKind.SELU -> 0.75
            NonLinearityKind.ExplicitGain -> explicitGainValue
        }

    override fun equals(other: Any?): Boolean =
        other is NonLinearity &&
            kind == other.kind &&
            (kind != NonLinearityKind.ExplicitGain || explicitGainValue == other.explicitGainValue)

    override fun hashCode(): Int = 31 * kind.hashCode() + explicitGainValue.hashCode()

    override fun toString(): String =
        if (kind == NonLinearityKind.ExplicitGain) {
            "ExplicitGain($explicitGainValue)"
        } else {
            kind.name
        }

    public companion object {
        public val ReLU: NonLinearity = NonLinearity(NonLinearityKind.ReLU)
        public val Linear: NonLinearity = NonLinearity(NonLinearityKind.Linear)
        public val Sigmoid: NonLinearity = NonLinearity(NonLinearityKind.Sigmoid)
        public val Tanh: NonLinearity = NonLinearity(NonLinearityKind.Tanh)
        public val SELU: NonLinearity = NonLinearity(NonLinearityKind.SELU)

        public fun ExplicitGain(g: Double): NonLinearity =
            NonLinearity(NonLinearityKind.ExplicitGain, g)
    }
}

public enum class InitKind {
    Const,
    Randn,
    Uniform,
    Kaiming,
}

/**
 * Variable initializations.
 */
public class Init(
    public val kind: InitKind = InitKind.Const,
    public val cst: Double = 0.0,
    public val mean: Double = 0.0,
    public val stdev: Double = 1.0,
    public val lo: Double = 0.0,
    public val up: Double = 1.0,
    public val dist: NormalOrUniform = NormalOrUniform.Uniform,
    public val fan: FanInOut = FanInOut.FanIn,
    public val nonLinearity: NonLinearity = NonLinearity.ReLU,
) {
    /**
     * Creates a new variable with the specified shape, device, and initialization.
     */
    public fun createVar(
        s: Shape,
        dtype: DType,
        device: Device,
    ): Var =
        when (kind) {
            InitKind.Const -> {
                if (cst == 0.0) {
                    Var.zeros(s, dtype, device)
                } else if (cst == 1.0) {
                    Var.ones(s, dtype, device)
                } else {
                    Var.fromTensor(Tensor.ones(s, dtype, device).affine(cst, 0.0))
                }
            }
            InitKind.Uniform -> Var.randF64(lo, up, s, dtype, device)
            InitKind.Randn -> Var.randnF64(mean, stdev, s, dtype, device)
            InitKind.Kaiming -> {
                val fanVal = fan.forShape(s)
                val gain = nonLinearity.gain()
                val std = gain / sqrt(fanVal.toDouble())
                when (dist) {
                    NormalOrUniform.Uniform -> {
                        val bound = sqrt(3.0) * std
                        Var.randF64(-bound, bound, s, dtype, device)
                    }
                    NormalOrUniform.Normal -> Var.randnF64(0.0, std, s, dtype, device)
                }
            }
        }

    public fun createVar(
        s: Int,
        dtype: DType,
        device: Device,
    ): Var = createVar(Shape(s), dtype, device)

    public fun createVar(
        s: Pair<Int, Int>,
        dtype: DType,
        device: Device,
    ): Var = createVar(Shape(s.first, s.second), dtype, device)

    public fun createVar(
        s: Triple<Int, Int, Int>,
        dtype: DType,
        device: Device,
    ): Var = createVar(Shape(s.first, s.second, s.third), dtype, device)

    public fun createVar(
        s: List<Int>,
        dtype: DType,
        device: Device,
    ): Var = createVar(Shape(s), dtype, device)

    override fun equals(other: Any?): Boolean =
        other is Init &&
            kind == other.kind &&
            cst == other.cst &&
            mean == other.mean &&
            stdev == other.stdev &&
            lo == other.lo &&
            up == other.up &&
            dist == other.dist &&
            fan == other.fan &&
            nonLinearity == other.nonLinearity

    override fun hashCode(): Int {
        var result = kind.hashCode()
        result = 31 * result + cst.hashCode()
        result = 31 * result + mean.hashCode()
        result = 31 * result + stdev.hashCode()
        result = 31 * result + lo.hashCode()
        result = 31 * result + up.hashCode()
        result = 31 * result + dist.hashCode()
        result = 31 * result + fan.hashCode()
        result = 31 * result + nonLinearity.hashCode()
        return result
    }

    public companion object {
        public fun default(): Init = Const(0.0)

        public fun Const(v: Double): Init = Init(kind = InitKind.Const, cst = v)

        public fun Randn(
            mean: Double,
            stdev: Double,
        ): Init = Init(kind = InitKind.Randn, mean = mean, stdev = stdev)

        public fun Uniform(
            lo: Double,
            up: Double,
        ): Init = Init(kind = InitKind.Uniform, lo = lo, up = up)

        public fun Kaiming(
            dist: NormalOrUniform,
            fan: FanInOut,
            nonLinearity: NonLinearity,
        ): Init =
            Init(
                kind = InitKind.Kaiming,
                dist = dist,
                fan = fan,
                nonLinearity = nonLinearity,
            )
    }
}

public val ZERO: Init = Init.Const(0.0)
public val ONE: Init = Init.Const(1.0)

public val DEFAULT_KAIMING_UNIFORM: Init =
    Init.Kaiming(
        dist = NormalOrUniform.Uniform,
        fan = FanInOut.FanIn,
        nonLinearity = NonLinearity.ReLU,
    )

public val DEFAULT_KAIMING_NORMAL: Init =
    Init.Kaiming(
        dist = NormalOrUniform.Normal,
        fan = FanInOut.FanIn,
        nonLinearity = NonLinearity.ReLU,
    )
