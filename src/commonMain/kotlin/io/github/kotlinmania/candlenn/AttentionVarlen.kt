// port-lint: source attention/varlen.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import kotlin.math.abs
import kotlin.math.max

/**
 * Unfused variable-length flash attention (no fused kernels).
 *
 * Input shapes (packed, no batch dim):
 * - [q]: `[totalQ, Hq, D]`
 * - [k]: `[totalK, Hk, D]`
 * - [v]: `[totalK, Hv, D]`
 * - [seqlensQ], [seqlensK]: `[B]` (u32 per-sequence lengths)
 *
 * Output shape: `[totalQ, Hq, D]`
 */
public fun flashAttnVarlenUnfused(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alibiSlopes: Tensor?,
    seqlensQ: Tensor,
    seqlensK: Tensor,
    maxSeqlenQ: Int,
    maxSeqlenK: Int,
    softmaxScale: Float,
    causal: Boolean,
    windowSizeLeft: Int? = null,
    windowSizeRight: Int? = null,
): Tensor {
    if (maxSeqlenQ < 0 || maxSeqlenK < 0) {
        bail("maxSeqlen must be non-negative")
    }
    val device = q.device()
    val (_, numHeads, headDim) = q.dims3()
    val (totalK, numKvHeads, _) = k.dims3()

    val (kExpanded, vExpanded) = if (numHeads != numKvHeads) {
        if (numHeads % numKvHeads != 0) {
            bail("Invalid GQA config: numHeads=$numHeads not divisible by numKvHeads=$numKvHeads")
        }
        val repeatFactor = numHeads / numKvHeads
        val kRep = k
            .reshape(Shape.fromDims(totalK, numKvHeads, 1, headDim))
            .broadcastAs(Shape.fromDims(totalK, numKvHeads, repeatFactor, headDim))
            .reshape(totalK, numHeads, headDim)
        val vRep = v
            .reshape(Shape.fromDims(totalK, numKvHeads, 1, headDim))
            .broadcastAs(Shape.fromDims(totalK, numKvHeads, repeatFactor, headDim))
            .reshape(totalK, numHeads, headDim)
        Pair(kRep, vRep)
    } else {
        Pair(k, v)
    }

    val batchSize = seqlensQ.dims()[0]
    val outputs = ArrayList<Tensor>()

    val seqlensQVec = seqlensQ.toVec1<UInt>()
    val seqlensKVec = seqlensK.toVec1<UInt>()
    val cumsumQ = IntArray(batchSize + 1)
    val cumsumK = IntArray(batchSize + 1)
    for (i in 0 until batchSize) {
        cumsumQ[i + 1] = cumsumQ[i] + seqlensQVec[i].toInt()
        cumsumK[i + 1] = cumsumK[i] + seqlensKVec[i].toInt()
    }

    for (i in 0 until batchSize) {
        val seqLenQ = seqlensQVec[i].toInt()
        val seqLenK = seqlensKVec[i].toInt()

        if (seqLenQ == 0 || seqLenK == 0) {
            continue
        }

        if (causal && seqLenK < seqLenQ) {
            bail("causal attention requires seqLenK >= seqLenQ (got k=$seqLenK, q=$seqLenQ) for batch index $i")
        }

        val startQ = cumsumQ[i]
        val startK = cumsumK[i]

        val qSeq = q.narrow(0, startQ, seqLenQ).transpose(0, 1).contiguous()
        val kSeq = kExpanded.narrow(0, startK, seqLenK).transpose(0, 1).contiguous()
        val vSeq = vExpanded.narrow(0, startK, seqLenK).transpose(0, 1).contiguous()

        val kSeqT = kSeq.transpose(1, 2).contiguous()
        var attentionScores = qSeq.matmul(kSeqT)

        val scaleTensor = Tensor.new(floatArrayOf(softmaxScale), Shape.SCALAR, device).toDtype(attentionScores.dtype())
        attentionScores = attentionScores.mul(scaleTensor.broadcastAs(attentionScores.shape()))

        if (causal) {
            val causalMask = createCausalMaskBatch(seqLenQ, seqLenK, numHeads, device)
                .toDtype(attentionScores.dtype())
            attentionScores = attentionScores.add(causalMask)
        }

        if (windowSizeLeft != null || windowSizeRight != null) {
            val windowMask = createWindowMaskBatch(
                seqLenQ,
                seqLenK,
                numHeads,
                windowSizeLeft,
                windowSizeRight,
                device,
            ).toDtype(attentionScores.dtype())
            attentionScores = attentionScores.add(windowMask)
        }

        if (alibiSlopes != null) {
            val alibiBias = createAlibiBiasBatch(
                seqLenQ,
                seqLenK,
                numHeads,
                alibiSlopes,
                causal,
                device,
            ).toDtype(attentionScores.dtype())
            attentionScores = attentionScores.add(alibiBias)
        }

        val attentionProbs = softmaxLastDim(attentionScores)
        val contextLayer = attentionProbs.matmul(vSeq)

        val seqOutput = contextLayer.transpose(0, 1)
        outputs.add(seqOutput)
    }

    if (outputs.isEmpty()) {
        return Tensor.zeros(Shape.fromDims(0, numHeads, headDim), q.dtype(), device)
    }

    return Tensor.cat(outputs, 0)
}

private fun createCausalMaskBatch(
    seqLenQ: Int,
    seqLenK: Int,
    numHeads: Int,
    device: Device,
): Tensor {
    val offset = seqLenK - seqLenQ
    val mask = ArrayList<Float>(seqLenQ * seqLenK)
    for (i in 0 until seqLenQ) {
        for (j in 0 until seqLenK) {
            if (j > i + offset) {
                mask.add(Float.NEGATIVE_INFINITY)
            } else {
                mask.add(0.0f)
            }
        }
    }
    val maskTensor = Tensor.fromVec(mask, Shape.fromDims(seqLenQ, seqLenK), device)
    return maskTensor.expand(Shape.fromDims(numHeads, seqLenQ, seqLenK))
}

private fun createWindowMaskBatch(
    seqLenQ: Int,
    seqLenK: Int,
    numHeads: Int,
    windowLeft: Int?,
    windowRight: Int?,
    device: Device,
): Tensor {
    val offset = seqLenK - seqLenQ
    val mask = ArrayList<Float>(seqLenQ * seqLenK)
    if (windowLeft != null && windowRight != null) {
        for (i in 0 until seqLenQ) {
            val iK = i + offset
            for (j in 0 until seqLenK) {
                val leftDist = max(iK - j, 0)
                val rightDist = max(j - iK, 0)
                if (leftDist > windowLeft || rightDist > windowRight) {
                    mask.add(Float.NEGATIVE_INFINITY)
                } else {
                    mask.add(0.0f)
                }
            }
        }
    } else if (windowLeft != null) {
        for (i in 0 until seqLenQ) {
            val iK = i + offset
            for (j in 0 until seqLenK) {
                if (j > iK) {
                    mask.add(Float.NEGATIVE_INFINITY)
                } else {
                    val dist = iK - j
                    if (dist > windowLeft) {
                        mask.add(Float.NEGATIVE_INFINITY)
                    } else {
                        mask.add(0.0f)
                    }
                }
            }
        }
    } else if (windowRight == null) {
        for (idx in 0 until seqLenQ * seqLenK) {
            mask.add(0.0f)
        }
    } else {
        bail("windowRight specified without windowLeft")
    }

    val maskTensor = Tensor.fromVec(mask, Shape.fromDims(seqLenQ, seqLenK), device)
    return maskTensor.expand(Shape.fromDims(numHeads, seqLenQ, seqLenK))
}

private fun createAlibiBiasBatch(
    seqLenQ: Int,
    seqLenK: Int,
    numHeads: Int,
    alibiSlopes: Tensor,
    causal: Boolean,
    device: Device,
): Tensor {
    val slopes = alibiSlopes.toVec1<Float>()
    if (slopes.size != numHeads) {
        bail("alibiSlopes has len ${slopes.size}, expected numHeads=$numHeads")
    }

    val offset = seqLenK - seqLenQ
    val headBiases = ArrayList<Tensor>(numHeads)
    for (slope in slopes) {
        val bias = ArrayList<Float>(seqLenQ * seqLenK)
        for (i in 0 until seqLenQ) {
            val iK = i + offset
            for (j in 0 until seqLenK) {
                val dist = if (causal) {
                    max(iK - j, 0).toFloat()
                } else {
                    abs(j - iK).toFloat()
                }
                bias.add(-slope * dist)
            }
        }
        headBiases.add(Tensor.fromVec(bias, Shape.fromDims(seqLenQ, seqLenK), device))
    }

    return Tensor.stack(headBiases, 0)
}
