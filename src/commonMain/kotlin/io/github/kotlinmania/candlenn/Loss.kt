// port-lint: source loss.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

/**
 * The negative log likelihood loss.
 *
 * Arguments:
 *
 * * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
 *   of categories. This is expected to contain log probabilities.
 * * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
 *
 * The resulting tensor is a scalar containing the average value over the batch.
 */
public fun nll(
    inp: Tensor,
    target: Tensor,
): Tensor {
    val bSz = target.dims1()
    return inp
        .gather(target.unsqueeze(1), D.Minus1)
        .sumAll()
        .affine(-1.0 / bSz.toDouble(), 0.0)
}

/**
 * The cross-entropy loss.
 *
 * Arguments:
 *
 * * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
 *   of categories. This is expected to contain raw logits.
 * * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
 *
 * The resulting tensor is a scalar containing the average value over the batch.
 */
public fun crossEntropy(
    inp: Tensor,
    target: Tensor,
): Tensor {
    if (inp.rank() != 2) {
        bail("cross_entropy expects an input tensor of rank 2")
    }
    val logSm = logSoftmax(inp, 1)
    return nll(logSm, target)
}

/**
 * The mean squared error loss.
 */
public fun mse(
    inp: Tensor,
    target: Tensor,
): Tensor = (inp - target).sqr().meanAll()

/**
 * The binary cross-entropy with logit loss.
 *
 * Arguments:
 *
 * * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
 *   of categories. This is expected to contain raw logits.
 * * [target]: The ground truth labels as a tensor of u32 of dimension `N, C`.
 *
 * The resulting tensor is a scalar containing the average value over the batch.
 */
public fun binaryCrossEntropyWithLogit(
    inp: Tensor,
    target: Tensor,
): Tensor {
    val inpSm = sigmoid(inp)
    val leftSide = target * inpSm.log()
    val rightSide = (target.affine(-1.0, 1.0)) * (inpSm.affine(-1.0, 1.0).log())
    val loss = leftSide + rightSide
    return loss.neg().meanAll()
}

/**
 * Computes the Huber loss between `inp` and `target`, with tunable parameter `delta`.
 *
 * For each value `x` in `error = inp - target`:
 * - `0.5 * x^2` if `|x| <= delta`
 * - `delta * (|x| - 0.5 * delta)` otherwise
 *
 * The resulting tensor is a scalar containing the average value over all elements.
 */
public fun huber(
    inp: Tensor,
    target: Tensor,
    delta: Double,
): Tensor {
    if (delta <= 0.0) {
        bail("huber expects delta > 0, got $delta")
    }
    val absDiff = (inp - target).abs()
    val mask = absDiff.le(Tensor.full(delta, Shape(listOf(1)), absDiff.device()).toDtype(absDiff.dtype()).broadcastAs(absDiff.shape()))
    val smallVal = absDiff.sqr().affine(0.5, 0.0)
    val largeVal = absDiff.affine(delta, -0.5 * delta * delta)
    return mask.whereCond(smallVal, largeVal).meanAll()
}
