// port-lint: source linear.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.math.sqrt

/**
 * A linear layer Applies a linear transformation to the incoming data `y = x@w.t() + b`.
 */
public class Linear(
    private val weight: Tensor,
    private val bias: Tensor?,
) : Module {
    public companion object {
        public fun new(
            weight: Tensor,
            bias: Tensor?,
        ): Linear = Linear(weight, bias)
    }

    public fun weight(): Tensor = weight

    public fun bias(): Tensor? = bias

    override fun forward(xs: Tensor): Tensor {
        // When possible, we avoid using a broadcast-matmul as it is much slower than the standard
        // matmul planar kernel on metal and cuda.
        val dims = xs.dims()
        val x =
            if (dims.size >= 3) {
                val bDims = dims.subList(0, dims.size - 1)
                val m = dims[dims.size - 1]
                val bSize = bDims.fold(1) { acc, v -> acc * v }
                val xs2d = xs.reshape(Shape(bSize, m))
                val w = weight.t()
                val out2d = xs2d.matmul(w)
                val n = out2d.dim(1)
                out2d.reshape(Shape(bDims + listOf(n)))
            } else {
                val w =
                    when (dims.size) {
                        1 -> weight.t()
                        2 -> weight.t()
                        else -> weight.broadcastLeft(Shape(dims.subList(0, dims.size - 2))).t()
                    }
                xs.matmul(w)
            }
        return when (bias) {
            null -> x
            else -> x.broadcastAdd(bias)
        }
    }
}

/**
 * Create or initialize a new linear layer.
 *
 * This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
 */
public fun linear(
    inDim: Int,
    outDim: Int,
    vb: VarBuilder,
): Linear {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws = vb.getWithHints(Shape(outDim, inDim), "weight", initWs)
    val bound = 1.0 / sqrt(inDim.toDouble())
    val initBs = Init.Uniform(-bound, bound)
    val bs = vb.getWithHints(Shape(outDim), "bias", initBs)
    return Linear.new(ws, bs)
}

/**
 * Create or initialize a new linear layer without biases.
 */
public fun linearNoBias(
    inDim: Int,
    outDim: Int,
    vb: VarBuilder,
): Linear {
    val initWs = DEFAULT_KAIMING_NORMAL
    val ws = vb.getWithHints(Shape(outDim, inDim), "weight", initWs)
    return Linear.new(ws, null)
}

public fun linearB(
    inDim: Int,
    outDim: Int,
    bias: Boolean,
    vb: VarBuilder,
): Linear =
    if (bias) {
        linear(inDim, outDim, vb)
    } else {
        linearNoBias(inDim, outDim, vb)
    }
