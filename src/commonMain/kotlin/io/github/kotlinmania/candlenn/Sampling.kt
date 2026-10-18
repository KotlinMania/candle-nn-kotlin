// port-lint: source sampling.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toIndex

/**
 * Sample according to the Gumbel-Softmax distribution.
 */
public fun gumbelSoftmax(
    logits: Tensor,
    temperature: Double,
    dim: Int,
): Tensor {
    val dimIdx = Shape.toIndex(dim, logits.shape(), "gumbel-softmax")
    return if (temperature <= 0.0) {
        logits.argmax(dimIdx)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        val logitsF32 = logits.toDtype(DType.F32)
        val minusG =
            logitsF32
                .randLike(1e-7, 0.999)
                .log()
                .neg()
                .log()
        if (temperature == 1.0) {
            (logitsF32 - minusG).argmax(dimIdx)
        } else {
            (logitsF32 + minusG * (-temperature)).argmax(dimIdx)
        }
    }
}

/**
 * Sample according to the Gumbel-Softmax distribution along the given dimension.
 */
public fun gumbelSoftmax(
    logits: Tensor,
    temperature: Double,
    dim: io.github.kotlinmania.candle.Dim,
): Tensor = gumbelSoftmax(logits, temperature, dim.toIndex(logits.shape(), "gumbel-softmax"))
