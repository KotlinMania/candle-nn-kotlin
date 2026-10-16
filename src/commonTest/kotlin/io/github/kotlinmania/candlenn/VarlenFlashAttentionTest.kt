// port-lint: tests tests/varlen_flash_attention.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class VarlenFlashAttentionTest {
    private enum class VarlenImpl {
        CpuFlash,
        Unfused,
    }

    private fun runVarlen(
        impl: VarlenImpl,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        alibiSlopes: Tensor?,
        seqlensQ: Tensor,
        seqlensK: Tensor,
        maxQ: Int,
        maxK: Int,
        softmaxScale: Float,
        causal: Boolean,
        windowLeft: Int? = null,
        windowRight: Int? = null,
    ): Tensor =
        when (impl) {
            VarlenImpl.CpuFlash -> {
                flashAttnVarlenCpu(
                    q,
                    k,
                    v,
                    alibiSlopes,
                    seqlensQ,
                    seqlensK,
                    maxQ,
                    maxK,
                    softmaxScale,
                    causal,
                    windowLeft,
                    windowRight,
                )
            }

            VarlenImpl.Unfused -> {
                flashAttnVarlenUnfused(
                    q,
                    k,
                    v,
                    alibiSlopes,
                    seqlensQ,
                    seqlensK,
                    maxQ,
                    maxK,
                    softmaxScale,
                    causal,
                    windowLeft,
                    windowRight,
                )
            }
        }

    private data class VarlenInputs(
        val q: Tensor,
        val k: Tensor,
        val v: Tensor,
        val seqlensQ: Tensor,
        val seqlensK: Tensor,
        val maxQ: Int,
        val maxK: Int,
    )

    private fun makeVarlenInputsPrefill(
        batchSize: Int,
        numHeads: Int,
        numKvHeads: Int,
        headDim: Int,
        maxSeq: Int,
        device: Device,
    ): VarlenInputs {
        val rng = Random(456L)
        val seqlens = UIntArray(batchSize)
        var total = 0
        var maxL = 0
        for (i in 0 until batchSize) {
            val l = rng.nextInt(4, maxSeq + 1)
            seqlens[i] = l.toUInt()
            total += l
            maxL = max(maxL, l)
        }
        val qData = List(total * numHeads * headDim) { rng.nextFloat() * 2.0f - 1.0f }
        val kData = List(total * numKvHeads * headDim) { rng.nextFloat() * 2.0f - 1.0f }
        val vData = List(total * numKvHeads * headDim) { rng.nextFloat() * 2.0f - 1.0f }

        val q = Tensor.fromVec(qData, Shape(total, numHeads, headDim), device)
        val k = Tensor.fromVec(kData, Shape(total, numKvHeads, headDim), device)
        val v = Tensor.fromVec(vData, Shape(total, numKvHeads, headDim), device)
        val seqlensQ = Tensor.new(seqlens, Shape(batchSize), device)
        val seqlensK = Tensor.new(seqlens, Shape(batchSize), device)

        return VarlenInputs(q, k, v, seqlensQ, seqlensK, maxL, maxL)
    }

    private fun assertClose(
        a: Tensor,
        b: Tensor,
        precision: DType,
        label: String,
    ) {
        val aVec = a.toDtype(DType.F32).flattenAll().toVec1<Float>()
        val bVec = b.toDtype(DType.F32).flattenAll().toVec1<Float>()
        var maxDiff = 0.0f
        for (i in aVec.indices) {
            maxDiff = max(maxDiff, abs(aVec[i] - bVec[i]))
        }
        val tol = if (precision == DType.F16) 5e-3f else 1e-4f
        assertTrue(maxDiff < tol, "$label: maxDiff $maxDiff >= $tol")
    }

    private fun checkFusedVsUnfused(
        batchSize: Int,
        numHeads: Int,
        numKvHeads: Int,
        headDim: Int,
        maxSeq: Int,
        causal: Boolean,
        precision: DType,
    ) {
        val inputs = makeVarlenInputsPrefill(batchSize, numHeads, numKvHeads, headDim, maxSeq, Device.Cpu)
        val q = inputs.q.toDtype(precision)
        val k = inputs.k.toDtype(precision)
        val v = inputs.v.toDtype(precision)
        val scale = 1.0f / sqrt(headDim.toFloat())

        val outFused =
            runVarlen(
                VarlenImpl.CpuFlash,
                q,
                k,
                v,
                null,
                inputs.seqlensQ,
                inputs.seqlensK,
                inputs.maxQ,
                inputs.maxK,
                scale,
                causal,
            )
        val outUnfused =
            runVarlen(
                VarlenImpl.Unfused,
                q,
                k,
                v,
                null,
                inputs.seqlensQ,
                inputs.seqlensK,
                inputs.maxQ,
                inputs.maxK,
                scale,
                causal,
            )
        assertClose(outFused, outUnfused, precision, "varlen causal=$causal precision=$precision")
    }

    @Test
    fun testFusedVsUnfusedNoncausalF32() {
        checkFusedVsUnfused(3, 4, 4, 16, 16, causal = false, precision = DType.F32)
    }

    @Test
    fun testFusedVsUnfusedCausalF32() {
        checkFusedVsUnfused(3, 4, 4, 16, 16, causal = true, precision = DType.F32)
    }

    @Test
    fun testFusedVsUnfusedNoncausalF16() {
        checkFusedVsUnfused(3, 4, 4, 16, 16, causal = false, precision = DType.F16)
    }

    @Test
    fun testFusedVsUnfusedCausalF16() {
        checkFusedVsUnfused(3, 4, 4, 16, 16, causal = true, precision = DType.F16)
    }

    @Test
    fun testPrefillGqaF32() {
        checkFusedVsUnfused(3, 6, 2, 16, 16, causal = false, precision = DType.F32)
    }

    @Test
    fun testPrefillGqaCausalF32() {
        checkFusedVsUnfused(3, 6, 2, 16, 16, causal = true, precision = DType.F32)
    }
}
