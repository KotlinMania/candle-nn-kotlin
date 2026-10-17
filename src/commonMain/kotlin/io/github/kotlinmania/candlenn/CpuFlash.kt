package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import kotlin.math.ceil
import kotlin.math.log2
import kotlin.math.pow
import io.github.kotlinmania.candlenn.attention.standard.runFlashAttnCpu as runStandardFlashAttnCpu

internal fun dotF32(
    a: FloatArray,
    aOffset: Int,
    b: FloatArray,
    bOffset: Int,
    len: Int,
): Float {
    var acc = 0.0f
    for (i in 0 until len) {
        acc += a[aOffset + i] * b[bOffset + i]
    }
    return acc
}

internal fun CpuStorage.toFloatArraySlice(startOffset: Int): FloatArray =
    when (this) {
        is CpuStorage.F32 -> {
            if (startOffset == 0) data else data.copyOfRange(startOffset, data.size)
        }

        else -> {
            val slice = asSlice()
            FloatArray(slice.size - startOffset) { i -> slice[startOffset + i].toFloat() }
        }
    }

/**
 * Flash attention with automatic dispatch.
 */
public fun flashAttn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    attnMask: AttnMask,
    maxBias: Float? = null,
    softcap: Float? = null,
): Tensor {
    val b = q.dims()[0]

    if (b > 1) {
        val dt = q.dtype()
        val varlenOk = (dt == DType.F32 || dt == DType.F16) && softcap == null
        val maskOk = attnMask.kind == AttnMaskKind.Causal || attnMask.kind == AttnMaskKind.None

        if (!varlenOk || !maskOk) {
            val maskDesc =
                when (attnMask.kind) {
                    AttnMaskKind.Causal -> "Causal"
                    AttnMaskKind.None -> "None"
                    AttnMaskKind.Mask -> "Mask(tensor)"
                }
            bail(
                "CPU flash attention with B>1 requires: f32/f16 dtype, no softcap, " +
                    "and Causal or None mask. Got B=$b, dtype=$dt, softcap=$softcap, mask=$maskDesc",
            )
        }

        return flashAttnViaVarlen(q, k, v, softmaxScale, attnMask, maxBias)
    }

    return when (attnMask.kind) {
        AttnMaskKind.Causal -> {
            runCausalAttnCpu(
                q,
                k,
                v,
                softmaxScale,
                attnMask.kvOffset(),
                maxBias,
                softcap,
            )
        }

        AttnMaskKind.None -> {
            runStandardFlashAttnCpu(
                q,
                k,
                v,
                null,
                softmaxScale,
                maxBias,
                softcap,
            )
        }

        AttnMaskKind.Mask -> {
            runStandardFlashAttnCpu(
                q,
                k,
                v,
                attnMask.maskTensor,
                softmaxScale,
                maxBias,
                softcap,
            )
        }
    }
}

private fun flashAttnViaVarlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    attnMask: AttnMask,
    maxBias: Float?,
): Tensor {
    val qDims = q.dims()
    val kDims = k.dims()
    val b = qDims[0]
    val sQ = qDims[1]
    val hQ = qDims[2]
    val d = qDims[3]
    val sKv = kDims[1]
    val hKv = kDims[2]

    val causal = attnMask.isCausal()

    val qPacked = q.contiguous().reshape(b * sQ, hQ, d)
    val kPacked = k.contiguous().reshape(b * sKv, hKv, d)
    val vPacked = v.contiguous().reshape(b * sKv, hKv, d)

    val device = q.device()
    val seqlensQ = Tensor.new(UIntArray(b) { sQ.toUInt() }, Shape.fromDims(b), device)
    val seqlensK = Tensor.new(UIntArray(b) { sKv.toUInt() }, Shape.fromDims(b), device)

    val alibiSlopes =
        if (maxBias != null && maxBias > 0.0f) {
            val exp2 = ceil(log2(hQ.toFloat())).toInt()
            val n2 = 1 shl exp2
            val slopes =
                List(hQ) { h ->
                    2.0f.pow(-maxBias * (h + 1).toFloat() / n2.toFloat())
                }
            Tensor.fromVec(slopes, Shape.fromDims(hQ), device)
        } else {
            null
        }

    val ctx =
        flashAttnVarlenCpu(
            qPacked,
            kPacked,
            vPacked,
            alibiSlopes,
            seqlensQ,
            seqlensK,
            sQ,
            sKv,
            softmaxScale,
            causal,
            null,
            null,
        )

    return ctx.reshape(Shape.fromDims(b, sQ, hQ, d)).transpose(1, 2).contiguous()
}
