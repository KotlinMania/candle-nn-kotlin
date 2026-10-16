// port-lint: source attention/cpu_flash/causal.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.log2
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.tanh

private fun vecFmaddScalar(
    acc: FloatArray,
    accOffset: Int,
    v: FloatArray,
    vOffset: Int,
    len: Int,
    w: Float,
) {
    for (i in 0 until len) {
        acc[accOffset + i] += v[vOffset + i] * w
    }
}

private fun vecScale(
    acc: FloatArray,
    s: Float,
) {
    for (i in acc.indices) {
        acc[i] *= s
    }
}

private fun vecAdd(
    acc: FloatArray,
    v: FloatArray,
    vOffset: Int,
    len: Int,
) {
    for (i in 0 until len) {
        acc[i] += v[vOffset + i]
    }
}

/**
 * Causal attention with loop-bound masking, B=1 only.
 */
public fun runCausalAttnCpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    kvOffset: Int,
    maxBias: Float? = null,
    softcap: Float? = null,
): Tensor {
    val b = q.dims()[0]
    if (b != 1) {
        bail(
            "causal runCausalAttnCpu is B=1 only (got B=$b). " +
                "Multi-batch should be routed through the varlen path.",
        )
    }

    val qSq = q.squeeze(0).contiguous()
    val kSq = k.squeeze(0).contiguous()
    val vSq = v.squeeze(0).contiguous()

    val (sQ, hQ, d) = qSq.dims3()
    val (sKv, hKv, _) = kSq.dims3()
    val (_, hV, _) = vSq.dims3()

    val maxBiasVal = maxBias ?: 0.0f
    val softcapVal = softcap ?: 0.0f

    val (qG, qL) = qSq.storageAndLayout()
    val qData =
        when (qG) {
            is Storage.Cpu -> qG.storage.toFloatArraySlice(qL.startOffset())
            else -> bail("Expected CPU storage")
        }
    val (kG, kL) = kSq.storageAndLayout()
    val kData =
        when (kG) {
            is Storage.Cpu -> kG.storage.toFloatArraySlice(kL.startOffset())
            else -> bail("Expected CPU storage")
        }
    val (vG, vL) = vSq.storageAndLayout()
    val vData =
        when (vG) {
            is Storage.Cpu -> vG.storage.toFloatArraySlice(vL.startOffset())
            else -> bail("Expected CPU storage")
        }

    return if (qSq.dtype() == DType.F32) {
        val result =
            if (sQ == 1) {
                causalDecodeF32(
                    qData,
                    kData,
                    vData,
                    hQ,
                    hKv,
                    hV,
                    d,
                    sKv,
                    softmaxScale,
                    maxBiasVal,
                    softcapVal,
                )
            } else {
                causalPrefillF32(
                    qData,
                    kData,
                    vData,
                    sQ,
                    hQ,
                    hKv,
                    hV,
                    d,
                    sKv,
                    softmaxScale,
                    kvOffset,
                    maxBiasVal,
                    softcapVal,
                )
            }
        result.unsqueeze(0)
    } else {
        val result =
            if (sQ == 1) {
                causalDecodeGeneric(
                    qData,
                    kData,
                    vData,
                    hQ,
                    hKv,
                    hV,
                    d,
                    sKv,
                    softmaxScale,
                    maxBiasVal,
                    softcapVal,
                )
            } else {
                causalPrefillGeneric(
                    qData,
                    kData,
                    vData,
                    sQ,
                    hQ,
                    hKv,
                    hV,
                    d,
                    sKv,
                    softmaxScale,
                    kvOffset,
                    maxBiasVal,
                    softcapVal,
                )
            }
        result.unsqueeze(0)
    }
}

private fun causalDecodeF32(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor {
    if (maxBias == 0.0f && logitSoftcap == 0.0f) {
        return causalDecodeF32Lean(qData, kData, vData, hQ, hKv, hV, d, kvLen, scale)
    }

    val rk = hQ / hKv
    val rv = hQ / hV
    val n2 = 1 shl ceil(log2(hQ.toFloat())).toInt()

    val scalePre = if (logitSoftcap != 0.0f) scale / logitSoftcap else scale
    val doSoftcap = logitSoftcap != 0.0f

    val kSeqStride = hKv * d
    val vSeqStride = hV * d

    val out = FloatArray(hQ * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (hI in 0 until hQ) {
        val slope = 2.0f.pow(-maxBias * (hI + 1).toFloat() / n2.toFloat())
        val kHeadOff = (hI / rk) * d
        val vHeadOff = (hI / rv) * d
        val qBase = hI * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val alibiBias = slope * (kvPos.toFloat() - (kvLen - 1).toFloat())
            val kBase = kvPos * kSeqStride + kHeadOff

            var score = dotF32(qData, qBase, kData, kBase, d) * scalePre
            if (doSoftcap) {
                score = logitSoftcap * tanh(score)
            }
            score += alibiBias

            val vBase = kvPos * vSeqStride + vHeadOff
            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += vData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        vecFmaddScalar(out, hI * d, acc, 0, d, inv)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, 1, d), Device.Cpu)
}

private fun causalDecodeF32Lean(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
): Tensor {
    val rk = hQ / hKv
    val rv = hQ / hV
    val kSeqStride = hKv * d
    val vSeqStride = hV * d

    val out = FloatArray(hQ * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (hI in 0 until hQ) {
        val kHeadOff = (hI / rk) * d
        val vHeadOff = (hI / rv) * d
        val qBase = hI * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val kBase = kvPos * kSeqStride + kHeadOff
            val score = dotF32(qData, qBase, kData, kBase, d) * scale
            val vBase = kvPos * vSeqStride + vHeadOff

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += vData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        vecFmaddScalar(out, hI * d, acc, 0, d, inv)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, 1, d), Device.Cpu)
}

/**
 * Decode with interleaved KV cache. No ALiBi, no softcap.
 */
public fun causalDecodeF32Interleaved(
    qData: FloatArray,
    kvData: FloatArray,
    hQ: Int,
    hKv: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
): Tensor {
    val rk = hQ / hKv
    if (rk == 2) {
        return causalDecodeF32InterleavedGqa2(qData, kvData, hKv, d, kvLen, scale)
    }
    val kvHeadStride = 2 * d
    val kvSeqStride = hKv * kvHeadStride

    val out = FloatArray(hQ * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (hI in 0 until hQ) {
        val kvHeadOff = (hI / rk) * kvHeadStride
        val qBase = hI * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val kvBase = kvPos * kvSeqStride + kvHeadOff
            val kBase = kvBase
            val vBase = kvBase + d

            val score = dotF32(qData, qBase, kvData, kBase, d) * scale
            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += kvData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        vecFmaddScalar(out, hI * d, acc, 0, d, inv)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, 1, d), Device.Cpu)
}

private fun causalDecodeF32InterleavedGqa2(
    qData: FloatArray,
    kvData: FloatArray,
    hKv: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
): Tensor {
    val hQ = hKv * 2
    val kvHeadStride = 2 * d
    val kvSeqStride = hKv * kvHeadStride

    val out = FloatArray(hQ * d)
    val acc0 = FloatArray(d)
    val acc1 = FloatArray(d)

    for (kvH in 0 until hKv) {
        val kvHeadOff = kvH * kvHeadStride
        val q0Base = kvH * 2 * d
        val q1Base = (kvH * 2 + 1) * d

        acc0.fill(0.0f)
        acc1.fill(0.0f)
        var m0 = Float.NEGATIVE_INFINITY
        var m1 = Float.NEGATIVE_INFINITY
        var s0 = 0.0f
        var s1 = 0.0f

        for (kvPos in 0 until kvLen) {
            val kvBase = kvPos * kvSeqStride + kvHeadOff
            val kBase = kvBase
            val vBase = kvBase + d

            val sc0 = dotF32(qData, q0Base, kvData, kBase, d) * scale
            val sc1 = dotF32(qData, q1Base, kvData, kBase, d) * scale

            if (sc0 > m0) {
                val so = exp(m0 - sc0)
                vecScale(acc0, so)
                s0 *= so
                m0 = sc0
                vecAdd(acc0, kvData, vBase, d)
                s0 += 1.0f
            } else {
                val w = exp(sc0 - m0)
                vecFmaddScalar(acc0, 0, kvData, vBase, d, w)
                s0 += w
            }

            if (sc1 > m1) {
                val so = exp(m1 - sc1)
                vecScale(acc1, so)
                s1 *= so
                m1 = sc1
                vecAdd(acc1, kvData, vBase, d)
                s1 += 1.0f
            } else {
                val w = exp(sc1 - m1)
                vecFmaddScalar(acc1, 0, kvData, vBase, d, w)
                s1 += w
            }
        }

        val inv0 = if (s0 > 0.0f) 1.0f / s0 else 0.0f
        val inv1 = if (s1 > 0.0f) 1.0f / s1 else 0.0f
        vecFmaddScalar(out, kvH * 2 * d, acc0, 0, d, inv0)
        vecFmaddScalar(out, (kvH * 2 + 1) * d, acc1, 0, d, inv1)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, 1, d), Device.Cpu)
}

private fun causalPrefillF32(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    sQ: Int,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
    kvOffset: Int,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor {
    if (maxBias == 0.0f && logitSoftcap == 0.0f) {
        return causalPrefillF32Lean(qData, kData, vData, sQ, hQ, hKv, hV, d, kvLen, scale, kvOffset)
    }

    val rk = hQ / hKv
    val rv = hQ / hV
    val n2 = 1 shl ceil(log2(hQ.toFloat())).toInt()

    val scalePre = if (logitSoftcap != 0.0f) scale / logitSoftcap else scale
    val doSoftcap = logitSoftcap != 0.0f

    val qSeqStride = hQ * d
    val kSeqStride = hKv * d
    val vSeqStride = hV * d

    val out = FloatArray(hQ * sQ * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (rowIdx in 0 until hQ * sQ) {
        val hI = rowIdx / sQ
        val qPos = rowIdx % sQ

        val slope =
            if (maxBias > 0.0f) {
                2.0f.pow(-maxBias * (hI + 1).toFloat() / n2.toFloat())
            } else {
                0.0f
            }

        val kHeadOff = (hI / rk) * d
        val vHeadOff = (hI / rv) * d
        val qBase = qPos * qSeqStride + hI * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        val kvEnd = min(qPos + kvOffset + 1, kvLen)

        for (kvPos in 0 until kvEnd) {
            val alibiBias =
                if (maxBias > 0.0f) {
                    slope * (kvPos.toLong() - (qPos + kvOffset).toLong()).toFloat()
                } else {
                    0.0f
                }

            val kBase = kvPos * kSeqStride + kHeadOff
            var score = dotF32(qData, qBase, kData, kBase, d) * scalePre
            if (doSoftcap) {
                score = logitSoftcap * tanh(score)
            }
            score += alibiBias

            val vBase = kvPos * vSeqStride + vHeadOff
            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += vData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        vecFmaddScalar(out, rowIdx * d, acc, 0, d, inv)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, sQ, d), Device.Cpu)
}

private fun causalPrefillF32Lean(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    sQ: Int,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
    kvOffset: Int,
): Tensor {
    val rk = hQ / hKv
    val rv = hQ / hV
    val qSeqStride = hQ * d
    val kSeqStride = hKv * d
    val vSeqStride = hV * d

    val out = FloatArray(hQ * sQ * d)
    val acc = FloatArray(d)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (rowIdx in 0 until hQ * sQ) {
        val hI = rowIdx / sQ
        val qPos = rowIdx % sQ

        val kHeadOff = (hI / rk) * d
        val vHeadOff = (hI / rv) * d
        val qBase = qPos * qSeqStride + hI * d

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        val kvEnd = min(qPos + kvOffset + 1, kvLen)

        for (kvPos in 0 until kvEnd) {
            val kBase = kvPos * kSeqStride + kHeadOff
            val score = dotF32(qData, qBase, kData, kBase, d) * scale
            val vBase = kvPos * vSeqStride + vHeadOff

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                for (t in 0 until d) {
                    accBuf[t] += vData[vBase + t] * w
                }
            }
        }

        val inv = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        vecFmaddScalar(out, rowIdx * d, acc, 0, d, inv)
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(hQ, sQ, d), Device.Cpu)
}

private fun causalDecodeGeneric(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor = causalDecodeF32(qData, kData, vData, hQ, hKv, hV, d, kvLen, scale, maxBias, logitSoftcap)

private fun causalPrefillGeneric(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    sQ: Int,
    hQ: Int,
    hKv: Int,
    hV: Int,
    d: Int,
    kvLen: Int,
    scale: Float,
    kvOffset: Int,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor = causalPrefillF32(qData, kData, vData, sQ, hQ, hKv, hV, d, kvLen, scale, kvOffset, maxBias, logitSoftcap)
