// port-lint: source attention/cpu_flash/varlen.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Float16Utils
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Fused variable-length flash attention on CPU.
 *
 * Input shapes (packed, no batch dim):
 * - [q]: `[totalQ, Hq, D]`
 * - [k]: `[totalK, Hk, D]`
 * - [v]: `[totalK, Hv, D]`
 * - [seqlensQ], [seqlensK]: `[B]` (u32 per-sequence lengths)
 *
 * Output shape: `[totalQ, Hq, D]`
 */
public fun flashAttnVarlenCpu(
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
    windowLeft: Int? = null,
    windowRight: Int? = null,
): Tensor {
    if (!q.device().isCpu()) {
        bail("flashAttnVarlenCpu is CPU only")
    }
    if (maxSeqlenQ < 0 || maxSeqlenK < 0) {
        bail("maxSeqlen must be non-negative")
    }

    val dt = q.dtype()
    if (k.dtype() != dt || v.dtype() != dt) {
        bail("q/k/v must have the same dtype (got q=$dt, k=${k.dtype()}, v=${v.dtype()})")
    }
    if (dt != DType.F32 && dt != DType.F16) {
        bail("flashAttnVarlenCpu supports f32 or f16 only")
    }

    val (totalQ, hq, d) = q.dims3()
    val (totalK, hk, dk) = k.dims3()
    val (_, hv, dv) = v.dims3()
    if (dk != d || dv != d) {
        bail("head dim mismatch")
    }

    if (hq % hk != 0) {
        bail("Invalid GQA: Hq=$hq not divisible by Hk=$hk")
    }
    if (hq % hv != 0) {
        bail("Invalid GQA: Hq=$hq not divisible by Hv=$hv")
    }
    val rk = hq / hk
    val rv = hq / hv

    val seqlensQVec = seqlensQ.toVec1<UInt>()
    val seqlensKVec = seqlensK.toVec1<UInt>()
    val bsz = seqlensQVec.size
    if (seqlensKVec.size != bsz) {
        bail("seqlensQ and seqlensK batch mismatch")
    }

    if (causal) {
        for (i in 0 until bsz) {
            val lq = seqlensQVec[i].toInt()
            val lk = seqlensKVec[i].toInt()
            if (lk < lq) {
                bail("causal requires lk>=lq, got lk=$lk lq=$lq at batch $i")
            }
        }
    }

    val cumsumQ = IntArray(bsz + 1)
    val cumsumK = IntArray(bsz + 1)
    for (i in 0 until bsz) {
        cumsumQ[i + 1] = cumsumQ[i] + seqlensQVec[i].toInt()
        cumsumK[i + 1] = cumsumK[i] + seqlensKVec[i].toInt()
    }
    if (cumsumQ[bsz] != totalQ || cumsumK[bsz] != totalK) {
        bail("totalQ/totalK mismatch with seqlens")
    }

    val slopes: List<Float>? = if (alibiSlopes != null) {
        val s = alibiSlopes.toVec1<Float>()
        if (s.size != hq) {
            bail("alibiSlopes len ${s.size} != Hq $hq")
        }
        s
    } else {
        null
    }

    val qCont = q.contiguous()
    val kCont = k.contiguous()
    val vCont = v.contiguous()

    val batchOfQ = IntArray(totalQ)
    val posInB = IntArray(totalQ)
    for (b in 0 until bsz) {
        val start = cumsumQ[b]
        val lq = seqlensQVec[b].toInt()
        for (t in 0 until lq) {
            batchOfQ[start + t] = b
            posInB[start + t] = t
        }
    }

    val (qG, qL) = qCont.storageAndLayout()
    val qData = when (qG) {
        is Storage.Cpu -> qG.storage.toFloatArraySlice(qL.startOffset())
        else -> bail("q not cpu")
    }
    val (kG, kL) = kCont.storageAndLayout()
    val kData = when (kG) {
        is Storage.Cpu -> kG.storage.toFloatArraySlice(kL.startOffset())
        else -> bail("k not cpu")
    }
    val (vG, vL) = vCont.storageAndLayout()
    val vData = when (vG) {
        is Storage.Cpu -> vG.storage.toFloatArraySlice(vL.startOffset())
        else -> bail("v not cpu")
    }

    val out = FloatArray(totalQ * hq * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (row in 0 until totalQ * hq) {
        val qIdx = row / hq
        val h = row % hq
        val outBase = row * d

        val b = batchOfQ[qIdx]
        val qPos = posInB[qIdx]

        val lq = seqlensQVec[b].toInt()
        val lk = seqlensKVec[b].toInt()
        if (lq == 0 || lk == 0) {
            continue
        }

        val startK = cumsumK[b]
        val offset = lk - lq

        val range = keyRange(qPos, lk, offset, causal, windowLeft, windowRight) ?: continue
        val j0 = range.first
        val j1 = range.second
        if (j0 > j1) {
            continue
        }

        val kHead = h / rk
        val vHead = h / rv
        val slope = slopes?.get(h) ?: 0.0f
        val iK = qPos + offset

        val qBase = (qIdx * hq + h) * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (j in j0..j1) {
            val kBase = ((startK + j) * hk + kHead) * d
            var score = dotF32(qData, qBase, kData, kBase, d) * softmaxScale
            if (slopes != null) {
                score += alibiBias(slope, iK, j, causal)
            }

            val vBase = ((startK + j) * hv + vHead) * d
            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += vData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        if (dt == DType.F16) {
            for (t in 0 until d) {
                val bits = Float16Utils.floatToF16Bits(acc[t] * inv)
                out[outBase + t] = Float16Utils.f16BitsToFloat(bits)
            }
        } else {
            for (t in 0 until d) {
                out[outBase + t] = acc[t] * inv
            }
        }
    }

    val result = Tensor.fromVec(out.toList(), Shape.fromDims(totalQ, hq, d), Device.Cpu)
    return if (dt == DType.F16) result.toDtype(DType.F16) else result
}

private fun alibiBias(slope: Float, iK: Int, j: Int, causal: Boolean): Float {
    val dist = if (causal) {
        max(iK - j, 0).toFloat()
    } else {
        abs(j - iK).toFloat()
    }
    return -slope * dist
}

private fun keyRange(
    qPos: Int,
    seqLenK: Int,
    offset: Int,
    causal: Boolean,
    wl: Int?,
    wr: Int?,
): Pair<Int, Int>? {
    if (seqLenK == 0) {
        return Pair(1, 0)
    }

    val lk = seqLenK
    val iK = qPos + offset

    var lo = 0
    var hi = lk - 1

    if (causal) {
        hi = min(hi, iK)
    }

    if (wl != null && wr != null) {
        lo = max(lo, iK - wl)
        hi = min(hi, iK + wr)
    } else if (wl != null) {
        hi = min(hi, iK)
        lo = max(lo, iK - wl)
    } else if (wr != null) {
        return null
    }

    lo = max(lo, 0)
    hi = min(hi, lk - 1)

    return if (lo > hi) Pair(1, 0) else Pair(lo, hi)
}
