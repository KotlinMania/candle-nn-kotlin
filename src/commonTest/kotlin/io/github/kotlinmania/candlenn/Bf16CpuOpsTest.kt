// port-lint: tests tests/bf16_cpu_ops.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Bf16CpuOpsTest {
    private fun rel(
        a: Tensor,
        b: Tensor,
    ): Float {
        val d =
            a
                .sub(b.toDtype(DType.F32))
                .abs()
                .maxAll()
                .toScalar<Float>()
        val s = max(a.abs().maxAll().toScalar<Float>(), 1.0f)
        return d / s
    }

    @Test
    fun bf16CpuOpSweep() {
        val dev = Device.Cpu
        val x32 = Tensor.rand(-2.0f, 2.0f, Shape(2, 4, 64), dev)
        val xb = x32.toDtype(DType.BF16)

        val checks =
            mutableListOf(
                "silu" to rel(silu(x32), silu(xb)),
                "softmax" to rel(softmaxLastDim(x32), softmaxLastDim(xb)),
                "mul" to rel(x32.mul(x32), xb.mul(xb)),
                "add" to rel(x32.add(x32), xb.add(xb)),
                "exp" to rel(x32.exp(), xb.exp()),
            )
        val w32 = Tensor.rand(0.5f, 1.5f, Shape(64), dev)
        val rms32 = rmsNorm(x32, w32, 1e-6f)
        val rmsb = rmsNorm(xb, w32.toDtype(DType.BF16), 1e-6f)
        checks.add("rmsNorm" to rel(rms32, rmsb))

        for ((name, r) in checks) {
            assertTrue(r < 3e-2f, "$name broken: rel $r")
        }
    }

    @Test
    fun bf16CpuMatmulWorks() {
        val dev = Device.Cpu
        val a = Tensor.rand(-1.0f, 1.0f, Shape(2, 16, 32), dev)
        val b = Tensor.rand(-1.0f, 1.0f, Shape(2, 32, 17), dev)
        val want = a.matmul(b)
        val got = a.toDtype(DType.BF16).matmul(b.toDtype(DType.BF16)).toDtype(DType.F32)
        val d =
            want
                .sub(got)
                .abs()
                .maxAll()
                .toScalar<Float>()
        assertTrue(d < 0.15f, "bf16 matmul diff $d")
    }

    @Test
    fun tileConfigThresholds() {
        assertEquals(16, defaultConfig(32, 4).bm)
        assertEquals(32, defaultConfig(33, 4).bm)
        assertEquals(64, defaultConfig(97, 4).bm)
        assertEquals(128, defaultConfig(513, 4).bm)
        assertEquals(64, defaultConfig(64, 4).bn)
        assertEquals(128, defaultConfig(65, 4).bn)
        assertEquals(128, defaultConfig(64, 4).bk)
        assertEquals(64, defaultConfig(65, 4).bk)
        assertEquals(1, defaultConfig(128, 1).groupM)
        assertEquals(16, defaultConfig(129, 1).groupM)
    }

    @Test
    fun sdpaCpuReferenceAgreement() {
        // Upstream tests/sdpa.rs tests Metal-specific SDPA kernels (Device::new_metal(0));
        // here we verify the CPU sdpa implementation against the reference matmul + softmax formula.
        val dev = Device.Cpu
        val bs = 2
        val h = 3
        val r = 4
        val l = 4
        val dk = 16
        val scale = 1.0f / kotlin.math.sqrt(dk.toFloat())
        val q = Tensor.randn(0.0f, 1.0f, Shape(bs, h, r, dk), dev)
        val k = Tensor.randn(0.0f, 1.0f, Shape(bs, h, l, dk), dev)
        val v = Tensor.randn(0.0f, 1.0f, Shape(bs, h, l, dk), dev)

        val att = q.mul(scale.toDouble()).matmul(k.t())
        val probs = softmaxLastDim(att)
        val groundTruth = probs.matmul(v)

        val sdpaOutput = sdpa(q, k, v, scale, 1.0f)
        assertEquals(groundTruth.shape(), sdpaOutput.shape())
        val err =
            groundTruth
                .sub(sdpaOutput)
                .abs()
                .sumAll()
                .toScalar<Float>()
        assertTrue(err <= 1e-4f, "sdpa error $err")
    }
}
