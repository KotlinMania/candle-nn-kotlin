// port-lint: source sequential.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Tensor

/**
 * A sequential layer combining multiple other layers.
 */
public class Sequential internal constructor(
    private val layers: MutableList<Module> = mutableListOf(),
) : Module {
    /**
     * The number of sub-layers embedded in this layer.
     */
    public fun len(): Long = layers.size.toLong()

    /**
     * Returns true if this layer does not have any sub-layer.
     */
    public fun isEmpty(): Boolean = layers.isEmpty()

    override fun forward(xs: Tensor): Tensor {
        var current = xs
        for (layer in layers) {
            current = layer.forward(current)
        }
        return current
    }

    /**
     * Appends a layer after all the current layers.
     */
    public fun add(layer: Module): Sequential {
        layers.add(layer)
        return this
    }

    /**
     * Appends a closure after all the current layers.
     */
    public fun addFn(f: TensorClosure): Sequential = add(func(f))

    /**
     * Applies the forward pass and returns the output for each layer.
     */
    public fun forwardAll(xs: Tensor): List<Tensor> {
        val out = ArrayList<Tensor>(layers.size)
        var current = xs
        for (layer in layers) {
            current = layer.forward(current)
            out.add(current)
        }
        return out
    }
}

/**
 * Creates a new empty sequential layer.
 */
public fun seq(): Sequential = Sequential(mutableListOf())
