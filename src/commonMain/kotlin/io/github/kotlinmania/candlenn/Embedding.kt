// port-lint: source embedding.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor

/**
 * A simple lookup table that stores embeddings of a fixed dictionary and size.
 */
public class Embedding(
    private val embeddings: Tensor,
    private val hiddenSize: Int,
) : Module {
    public companion object {
        public fun new(
            embeddings: Tensor,
            hiddenSize: Int,
        ): Embedding = Embedding(embeddings, hiddenSize)
    }

    public fun embeddings(): Tensor = embeddings

    /**
     * Get the hidden size of the embedding matrix
     */
    public fun hiddenSize(): Int = hiddenSize

    override fun forward(xs: Tensor): Tensor {
        val finalDims = xs.dims().toMutableList()
        finalDims.add(hiddenSize)
        val indexes = xs.flattenAll()
        val values = embeddings.indexSelect(indexes, 0)
        return values.reshape(Shape(finalDims))
    }
}

public fun embedding(
    inSize: Int,
    outSize: Int,
    vb: VarBuilder,
): Embedding {
    val embeddings =
        vb.getWithHints(
            Shape(inSize, outSize),
            "weight",
            Init.Randn(mean = 0.0, stdev = 1.0),
        )
    return Embedding.new(embeddings, outSize)
}
