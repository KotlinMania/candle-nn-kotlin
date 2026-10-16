// port-lint: source func.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.ModuleT
import io.github.kotlinmania.candle.Tensor

public fun interface TensorClosure {
    public fun invoke(xs: Tensor): Tensor
}

public fun interface TensorTrainClosure {
    public fun invoke(
        xs: Tensor,
        train: Boolean,
    ): Tensor
}

/**
 * A layer defined by a simple closure.
 */
public class Func(
    private val f: TensorClosure,
) : Module {
    override fun forward(xs: Tensor): Tensor = f.invoke(xs)

    public fun clone(): Func = Func(f)

    override fun toString(): String = "func"

    public companion object {
        public fun new(f: TensorClosure): Func = Func(f)
    }
}

public fun func(f: TensorClosure): Func = Func(f)

/**
 * A layer defined by a simple closure with a train flag.
 */
public class FuncT(
    private val f: TensorTrainClosure,
) : ModuleT {
    override fun forwardT(
        xs: Tensor,
        train: Boolean,
    ): Tensor = f.invoke(xs, train)

    public fun clone(): FuncT = FuncT(f)

    override fun toString(): String = "func"

    public companion object {
        public fun new(f: TensorTrainClosure): FuncT = FuncT(f)
    }
}

public fun funcT(f: TensorTrainClosure): FuncT = FuncT(f)
