// port-lint: tests tests/cpu_flash_attn.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

class CpuFlashAttnTest {
    private fun referenceSdpa(q: Tensor, k: Tensor, v: Tensor, scale: Float): Tensor {
        val att = q.mul(scale.toDouble()).matmul(k.t())
        val probs = softmaxLastDim(att.toDtype(DType.F32)).toDtype(q.dtype())
        return probs.matmul(v)
    }

    private fun referenceCausalSdpa(q: Tensor, k: Tensor, v: Tensor, scale: Float): Tensor {
        val (_, _, sQ, _) = q.dims4()
        val (_, _, sKv, _) = k.dims4()
        val offset = sKv - sQ

        val maskVals = ArrayList<Float>(sQ * sKv)
        for (i in 0 until sQ) {
            for (j in 0 until sKv) {
                if (j <= i + offset) {
                    maskVals.add(0.0f)
                } else {
                    maskVals.add(Float.NEGATIVE_INFINITY)
                }
            }
        }
        val mask = Tensor.fromVec(maskVals, Shape(1, 1, sQ, sKv), q.device())

        val att = q.mul(scale.toDouble()).matmul(k.t()).broadcastAdd(mask)
        val probs = softmaxLastDim(att.toDtype(DType.F32)).toDtype(q.dtype())
        return probs.matmul(v)
    }

    private fun assertClose(a: Tensor, b: Tensor, tol: Float, label: String) {
        val aFlat = a.toDtype(DType.F32).flattenAll().toVec1<Float>()
        val bFlat = b.toDtype(DType.F32).flattenAll().toVec1<Float>()
        var maxDiff = 0.0f
        for (i in aFlat.indices) {
            maxDiff = max(maxDiff, abs(aFlat[i] - bFlat[i]))
        }
        assertTrue(maxDiff < tol, "$label: max diff $maxDiff > tol $tol")
    }

    @Test
    fun causalDecodeB1() {
        val h = 4
        val d = 16
        val kvLen = 32
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, h, 1, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, h, kvLen, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, h, kvLen, d), dev)

        val expected = referenceSdpa(q, k, v, scale)
        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.causalWithOffset(kvLen - 1),
            null,
            null,
        )
        assertClose(out, expected, 1e-5f, "causal_decode_b1")
    }

    @Test
    fun causalPrefillB1() {
        val h = 4
        val d = 16
        val s = 8
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)

        val expected = referenceCausalSdpa(q, k, v, scale)
        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.causal(),
            null,
            null,
        )
        assertClose(out, expected, 1e-5f, "causal_prefill_b1")
    }

    @Test
    fun causalPrefillWithOffset() {
        val h = 4
        val d = 16
        val sQ = 4
        val sKv = 12
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu
        val offset = sKv - sQ

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, h, sQ, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, h, sKv, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, h, sKv, d), dev)

        val expected = referenceCausalSdpa(q, k, v, scale)
        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.causalWithOffset(offset),
            null,
            null,
        )
        assertClose(out, expected, 1e-5f, "causal_prefill_with_offset")
    }

    @Test
    fun causalDecodeGqa() {
        val hQ = 8
        val hKv = 2
        val d = 16
        val kvLen = 20
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, hQ, 1, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, hKv, kvLen, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, hKv, kvLen, d), dev)

        val reps = hQ / hKv
        val kExp = k
            .reshape(Shape(1, hKv, 1, kvLen, d))
            .broadcastAs(Shape(1, hKv, reps, kvLen, d))
            .reshape(Shape(1, hQ, kvLen, d))
        val vExp = v
            .reshape(Shape(1, hKv, 1, kvLen, d))
            .broadcastAs(Shape(1, hKv, reps, kvLen, d))
            .reshape(Shape(1, hQ, kvLen, d))

        val expected = referenceSdpa(q, kExp, vExp, scale)
        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.causalWithOffset(kvLen - 1),
            null,
            null,
        )
        assertClose(out, expected, 1e-5f, "causal_decode_gqa")
    }

    @Test
    fun standardNoMaskB1() {
        val h = 4
        val d = 16
        val s = 8
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)

        val expected = referenceSdpa(q, k, v, scale)
        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.None,
            null,
            null,
        )
        assertClose(out, expected, 1e-5f, "standard_no_mask_b1")
    }

    @Test
    fun standardWithMaskB1() {
        val h = 2
        val d = 8
        val s = 4
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, h, s, d), dev)

        val maskVals = ArrayList<Float>(s * s)
        for (i in 0 until s) {
            for (j in 0 until s) {
                maskVals.add(if (j <= i) 0.0f else Float.NEGATIVE_INFINITY)
            }
        }
        val maskTensor = Tensor.fromVec(maskVals, Shape(1, s, s), dev)

        val att = q.mul(scale.toDouble()).matmul(k.t()).broadcastAdd(maskTensor.unsqueeze(0))
        val probs = softmaxLastDim(att)
        val expected = probs.matmul(v)

        val out = flashAttn(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale,
            AttnMask.Mask(maskTensor),
            null,
            null,
        )
        assertClose(out, expected, 1e-4f, "standard_with_mask_b1")
    }

    @Test
    fun interleavedKvDecode() {
        val hQ = 8
        val hKv = 2
        val d = 16
        val kvLen = 20
        val scale = 1.0f / sqrt(d.toFloat())
        val dev = Device.Cpu

        val q = Tensor.randn(0.0f, 1.0f, Shape(1, hQ, 1, d), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(1, hKv, kvLen, d), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(1, hKv, kvLen, d), dev)

        val reps = hQ / hKv
        val kExp = k
            .reshape(Shape(1, hKv, 1, kvLen, d))
            .broadcastAs(Shape(1, hKv, reps, kvLen, d))
            .reshape(Shape(1, hQ, kvLen, d))
        val vExp = v
            .reshape(Shape(1, hKv, 1, kvLen, d))
            .broadcastAs(Shape(1, hKv, reps, kvLen, d))
            .reshape(Shape(1, hQ, kvLen, d))
        val expected = referenceSdpa(q, kExp, vExp, scale)

        val kSeq = k.squeeze(0).transpose(0, 1).contiguous()
        val vSeq = v.squeeze(0).transpose(0, 1).contiguous()
        val kv = Tensor.cat(listOf(kSeq, vSeq), 2).contiguous()

        val kvData = kv.flattenAll().toVec1<Float>().toFloatArray()
        val qFlat = q.squeeze(0).squeeze(1).contiguous().flattenAll().toVec1<Float>().toFloatArray()

        val out = causalDecodeF32Interleaved(qFlat, kvData, hQ, hKv, d, kvLen, scale).unsqueeze(0)
        assertClose(out, expected, 1e-5f, "interleaved_kv_decode")
    }
}
