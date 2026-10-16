// port-lint: source attention/cpu_flash/standard.rs
package io.github.kotlinmania.candlenn.attention.standard

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candlenn.dotF32
import io.github.kotlinmania.candlenn.onlineSoftmaxStep
import io.github.kotlinmania.candlenn.toFloatArraySlice
import kotlin.math.ceil
import kotlin.math.log2
import kotlin.math.pow
import kotlin.math.tanh

/**
 * Fused softmax(qk^T*scale)v on CPU, B=1 only. Output shape (1, H, S, Dv).
 */
public fun runFlashAttnCpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor?,
    softmaxScale: Float,
    maxBias: Float? = null,
    softcap: Float? = null,
): Tensor {
    val b = q.shape().dims()[0]
    if (b != 1) {
        bail(
            "standard runFlashAttnCpu is B=1 only (got B=$b). " +
                "Multi-batch should be routed through the varlen path.",
        )
    }

    val (qGuard, qLayout) = q.storageAndLayout()
    val qData =
        when (qGuard) {
            is Storage.Cpu -> qGuard.storage.toFloatArraySlice(qLayout.startOffset())
            else -> bail("Expected CPU storage for q")
        }
    val (kGuard, kLayout) = k.storageAndLayout()
    val kData =
        when (kGuard) {
            is Storage.Cpu -> kGuard.storage.toFloatArraySlice(kLayout.startOffset())
            else -> bail("Expected CPU storage for k")
        }
    val (vGuard, vLayout) = v.storageAndLayout()
    val vData =
        when (vGuard) {
            is Storage.Cpu -> vGuard.storage.toFloatArraySlice(vLayout.startOffset())
            else -> bail("Expected CPU storage for v")
        }
    val maskCont = mask?.contiguous()
    val maskData =
        if (maskCont != null) {
            val (mGuard, mLayout) = maskCont.storageAndLayout()
            when (mGuard) {
                is Storage.Cpu -> mGuard.storage.toFloatArraySlice(mLayout.startOffset())
                else -> bail("Expected CPU storage for mask")
            }
        } else {
            null
        }

    val qStride = q.stride()
    val kStride = k.stride()
    val vStride = v.stride()

    val maxBiasVal = maxBias ?: 0.0f
    val logitSoftcap = softcap ?: 0.0f
    val lean = maskData == null && maxBiasVal == 0.0f && logitSoftcap == 0.0f

    if (q.shape().dims()[1] == 1) {
        if (lean) {
            return flashAttnDecodeLean(
                qData,
                kData,
                vData,
                q.shape().dims(),
                k.shape().dims(),
                v.shape().dims(),
                qStride,
                kStride,
                vStride,
                softmaxScale,
            )
        }
        return flashAttnDecode(
            qData,
            kData,
            vData,
            maskData,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            qStride,
            kStride,
            vStride,
            softmaxScale,
            maxBiasVal,
            logitSoftcap,
        )
    }

    if (lean) {
        return flashAttnPrefillLean(
            qData,
            kData,
            vData,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            qStride,
            kStride,
            vStride,
            softmaxScale,
        )
    }
    return flashAttnPrefill(
        qData,
        kData,
        vData,
        maskData,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        qStride,
        kStride,
        vStride,
        softmaxScale,
        maxBiasVal,
        logitSoftcap,
    )
}

private fun flashAttnDecodeLean(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    qshape: List<Int>,
    kshape: List<Int>,
    vshape: List<Int>,
    qstride: List<Int>,
    kstride: List<Int>,
    vstride: List<Int>,
    scale: Float,
): Tensor {
    val h = qshape[2]
    val d = qshape[3]
    val kvLen = kshape[1]
    val kH = kshape[2]
    val vH = vshape[2]
    val rk2 = h / kH
    val rv2 = h / vH
    val dv = d
    val vContiguous = vstride[3] == 1

    val out = FloatArray(h * dv)
    val acc = FloatArray(dv)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (hI in 0 until h) {
        val kHead = hI / rk2
        val vHead = hI / rv2
        val qBase = hI * qstride[2]

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val kBase = kvPos * kstride[1] + kHead * kstride[2]
            val score = dotF32(qData, qBase, kData, kBase, d) * scale
            val vBase = kvPos * vstride[1] + vHead * vstride[2]

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                if (vContiguous) {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI] * w
                    }
                } else {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI * vstride[3]] * w
                    }
                }
            }
        }

        val invS = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        val outBase = hI * dv
        for (dI in 0 until dv) {
            out[outBase + dI] = acc[dI] * invS
        }
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(1, h, 1, dv), Device.Cpu)
}

private fun flashAttnDecode(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    maskVec: FloatArray?,
    qshape: List<Int>,
    kshape: List<Int>,
    vshape: List<Int>,
    qstride: List<Int>,
    kstride: List<Int>,
    vstride: List<Int>,
    scale: Float,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor {
    val h = qshape[2]
    val d = qshape[3]
    val kvLen = kshape[1]
    val kH = kshape[2]
    val vH = vshape[2]
    val rk2 = h / kH
    val rv2 = h / vH
    val dv = d

    val n2 = 1 shl ceil(log2(h.toFloat())).toInt()
    val scalePre = if (logitSoftcap != 0.0f) scale / logitSoftcap else scale
    val doSoftcap = logitSoftcap != 0.0f
    val vContiguous = vstride[3] == 1

    val out = FloatArray(h * dv)
    val acc = FloatArray(dv)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (hI in 0 until h) {
        val slope =
            if (maxBias > 0.0f) {
                2.0f.pow(-maxBias * (hI + 1).toFloat() / n2.toFloat())
            } else {
                1.0f
            }

        val kHead = hI / rk2
        val vHead = hI / rv2
        val qBase = hI * qstride[2]

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val mv =
                if (maskVec != null) {
                    slope * maskVec[kvPos]
                } else {
                    0.0f
                }
            if (mv == Float.NEGATIVE_INFINITY) {
                continue
            }

            val kBase = kvPos * kstride[1] + kHead * kstride[2]
            var score = dotF32(qData, qBase, kData, kBase, d) * scalePre
            if (doSoftcap) {
                score = logitSoftcap * tanh(score)
            }
            score += mv

            val vBase = kvPos * vstride[1] + vHead * vstride[2]

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                if (vContiguous) {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI] * w
                    }
                } else {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI * vstride[3]] * w
                    }
                }
            }
        }

        val invS = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        val outBase = hI * dv
        for (dI in 0 until dv) {
            out[outBase + dI] = acc[dI] * invS
        }
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(1, h, 1, dv), Device.Cpu)
}

private fun flashAttnPrefillLean(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    qshape: List<Int>,
    kshape: List<Int>,
    vshape: List<Int>,
    qstride: List<Int>,
    kstride: List<Int>,
    vstride: List<Int>,
    scale: Float,
): Tensor {
    val qLen = qshape[1]
    val h = qshape[2]
    val d = qshape[3]
    val kvLen = kshape[1]
    val kH = kshape[2]
    val vH = vshape[2]
    val rk2 = h / kH
    val rv2 = h / vH
    val dv = d
    val vContiguous = vstride[3] == 1

    val out = FloatArray(h * qLen * dv)
    val acc = FloatArray(dv)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (rowIdx in 0 until h * qLen) {
        val hI = rowIdx / qLen
        val qPos = rowIdx % qLen
        val kHead = hI / rk2
        val vHead = hI / rv2
        val qBase = qPos * qstride[1] + hI * qstride[2]

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val kBase = kvPos * kstride[1] + kHead * kstride[2]
            val score = dotF32(qData, qBase, kData, kBase, d) * scale
            val vBase = kvPos * vstride[1] + vHead * vstride[2]

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                if (vContiguous) {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI] * w
                    }
                } else {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI * vstride[3]] * w
                    }
                }
            }
        }

        val invS = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        val outBase = rowIdx * dv
        for (dI in 0 until dv) {
            out[outBase + dI] = acc[dI] * invS
        }
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(1, h, qLen, dv), Device.Cpu)
}

private fun flashAttnPrefill(
    qData: FloatArray,
    kData: FloatArray,
    vData: FloatArray,
    maskVec: FloatArray?,
    qshape: List<Int>,
    kshape: List<Int>,
    vshape: List<Int>,
    qstride: List<Int>,
    kstride: List<Int>,
    vstride: List<Int>,
    scale: Float,
    maxBias: Float,
    logitSoftcap: Float,
): Tensor {
    val qLen = qshape[1]
    val h = qshape[2]
    val d = qshape[3]
    val kvLen = kshape[1]
    val kH = kshape[2]
    val vH = vshape[2]
    val rk2 = h / kH
    val rv2 = h / vH
    val dv = d

    val n2 = 1 shl ceil(log2(h.toFloat())).toInt()
    val scalePre = if (logitSoftcap != 0.0f) scale / logitSoftcap else scale
    val doSoftcap = logitSoftcap != 0.0f
    val vContiguous = vstride[3] == 1

    val out = FloatArray(h * qLen * dv)
    val acc = FloatArray(dv)
    val m = FloatArray(1)
    val ssum = FloatArray(1)

    for (rowIdx in 0 until h * qLen) {
        val hI = rowIdx / qLen
        val qPos = rowIdx % qLen
        val slope =
            if (maxBias > 0.0f) {
                2.0f.pow(-maxBias * (hI + 1).toFloat() / n2.toFloat())
            } else {
                1.0f
            }

        val kHead = hI / rk2
        val vHead = hI / rv2
        val qBase = qPos * qstride[1] + hI * qstride[2]

        acc.fill(0.0f)
        m[0] = Float.NEGATIVE_INFINITY
        ssum[0] = 0.0f

        for (kvPos in 0 until kvLen) {
            val mv =
                if (maskVec != null) {
                    slope * maskVec[qPos * kvLen + kvPos]
                } else {
                    0.0f
                }
            if (mv == Float.NEGATIVE_INFINITY) {
                continue
            }

            val kBase = kvPos * kstride[1] + kHead * kstride[2]
            var score = dotF32(qData, qBase, kData, kBase, d) * scalePre
            if (doSoftcap) {
                score = logitSoftcap * tanh(score)
            }
            score += mv

            val vBase = kvPos * vstride[1] + vHead * vstride[2]

            onlineSoftmaxStep(score, m, ssum, acc) { accBuf, w ->
                if (vContiguous) {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI] * w
                    }
                } else {
                    for (dI in 0 until dv) {
                        accBuf[dI] += vData[vBase + dI * vstride[3]] * w
                    }
                }
            }
        }

        val invS = if (ssum[0] > 0.0f) 1.0f / ssum[0] else 0.0f
        val outBase = rowIdx * dv
        for (dI in 0 until dv) {
            out[outBase + dI] = acc[dI] * invS
        }
    }

    return Tensor.fromVec(out.toList(), Shape.fromDims(1, h, qLen, dv), Device.Cpu)
}
