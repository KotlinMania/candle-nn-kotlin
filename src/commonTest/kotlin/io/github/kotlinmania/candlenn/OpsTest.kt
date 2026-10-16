// port-lint: tests tests/ops.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toVec3Round
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class OpsTest {
    @Test
    fun softmaxCpu() {
        val device = Device.Cpu
        val data =
            floatArrayOf(
                3.0f,
                1.0f,
                4.0f,
                1.0f,
                5.0f,
                9.0f,
                2.0f,
                1.0f,
                7.0f,
                8.0f,
                2.0f,
                8.0f,
            )
        val tensor = Tensor.new(data, Shape(2, 2, 3), device)
        val t0 = softmax(tensor.log(), 0)
        val t1 = softmax(tensor.log(), 1)
        val t2 = softmax(tensor.log(), 2)
        assertEquals(
            listOf(
                listOf(listOf(0.6f, 0.5f, 0.3636f), listOf(0.1111f, 0.7143f, 0.5294f)),
                listOf(listOf(0.4f, 0.5f, 0.6364f), listOf(0.8889f, 0.2857f, 0.4706f)),
            ),
            toVec3Round(t0, 4),
        )
        assertEquals(
            listOf(
                listOf(listOf(0.75f, 0.1667f, 0.3077f), listOf(0.25f, 0.8333f, 0.6923f)),
                listOf(listOf(0.2f, 0.3333f, 0.4667f), listOf(0.8f, 0.6667f, 0.5333f)),
            ),
            toVec3Round(t1, 4),
        )
        assertEquals(
            listOf(
                listOf(listOf(0.375f, 0.125f, 0.5f), listOf(0.0667f, 0.3333f, 0.6f)),
                listOf(listOf(0.2f, 0.1f, 0.7f), listOf(0.4444f, 0.1111f, 0.4444f)),
            ),
            toVec3Round(t2, 4),
        )
        val tLast = softmaxLastDim(tensor.log())
        assertEquals(
            listOf(
                listOf(listOf(0.375f, 0.125f, 0.5f), listOf(0.0667f, 0.3333f, 0.6f)),
                listOf(listOf(0.2f, 0.1f, 0.7f), listOf(0.4444f, 0.1111f, 0.4444f)),
            ),
            toVec3Round(tLast, 4),
        )
    }

    @Test
    fun rmsNormCpu() {
        val device = Device.Cpu
        val data =
            floatArrayOf(
                3.0f,
                1.0f,
                4.0f,
                1.0f,
                5.0f,
                9.0f,
                2.0f,
                1.0f,
                7.0f,
                8.0f,
                2.0f,
                8.0f,
            )
        val tensor = Tensor.new(data, Shape(2, 2, 3), device)
        val alpha = Tensor.new(floatArrayOf(1.0f, 2.0f, 3.0f), Shape(3), device)
        val t = rmsNorm(tensor, alpha, 1e-5f)
        assertEquals(
            listOf(
                listOf(listOf(1.019f, 0.6794f, 4.0762f), listOf(0.1674f, 1.6744f, 4.521f)),
                listOf(listOf(0.4714f, 0.4714f, 4.9497f), listOf(1.206f, 0.603f, 3.6181f)),
            ),
            toVec3Round(t, 4),
        )
        val t2 = rmsNormSlow(tensor, alpha, 1e-5f)
        assertEquals(
            listOf(
                listOf(listOf(1.019f, 0.6794f, 4.0762f), listOf(0.1674f, 1.6744f, 4.521f)),
                listOf(listOf(0.4714f, 0.4714f, 4.9497f), listOf(1.206f, 0.603f, 3.6181f)),
            ),
            toVec3Round(t2, 4),
        )
        val diff =
            t
                .sub(t2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(diff < 1e-5f)
    }

    @Test
    fun rmsNormLargeMagnitudeCpu() {
        val device = Device.Cpu
        val rows = 4
        val hidden = 512
        val data =
            List(rows * hidden) { i ->
                val sign = if (i % 2 == 0) 1.0f else -1.0f
                sign * (abs(sin(i.toFloat() * 0.17f)) * 7e9f)
            }
        val tensor = Tensor.fromVec(data, Shape(rows, hidden), device)
        val alpha = Tensor.ones(Shape(hidden), DType.F32, device)

        val fused = rmsNorm(tensor, alpha, 1e-5f)
        val slow = rmsNormSlow(tensor, alpha, 1e-5f)

        val fusedV = fused.flattenAll().toVec1<Float>()
        val slowV = slow.flattenAll().toVec1<Float>()
        for (v in fusedV) {
            assertTrue(v.isFinite())
        }
        for (v in slowV) {
            assertTrue(v.isFinite())
        }
        var maxDiff = 0.0f
        for (i in fusedV.indices) {
            maxDiff = max(maxDiff, abs(fusedV[i] - slowV[i]))
        }
        assertTrue(maxDiff < 5e-3f)
    }

    @Test
    fun layerNormOpCpu() {
        val device = Device.Cpu
        val data =
            floatArrayOf(
                3.0f,
                1.0f,
                4.0f,
                1.0f,
                5.0f,
                9.0f,
                2.0f,
                1.0f,
                7.0f,
                8.0f,
                2.0f,
                8.0f,
            )
        val tensor = Tensor.new(data, Shape(2, 2, 3), device)
        val alpha = Tensor.new(floatArrayOf(1.0f, 2.0f, 3.0f), Shape(3), device)
        val beta = Tensor.new(floatArrayOf(0.5f, 0.0f, -0.2f), Shape(3), device)
        val t = layerNorm(tensor, alpha, beta, 1e-5f)
        assertEquals(
            listOf(
                listOf(listOf(0.7673f, -2.6726f, 3.0071f), listOf(-0.7247f, 0.0f, 3.4742f)),
                listOf(listOf(-0.008f, -1.778f, 3.991f), listOf(1.2071f, -2.8284f, 1.9213f)),
            ),
            toVec3Round(t, 4),
        )
        val t2 = layerNormSlow(tensor, alpha, beta, 1e-5f)
        assertEquals(
            listOf(
                listOf(listOf(0.7673f, -2.6726f, 3.0071f), listOf(-0.7247f, 0.0f, 3.4742f)),
                listOf(listOf(-0.008f, -1.778f, 3.991f), listOf(1.2071f, -2.8284f, 1.9213f)),
            ),
            toVec3Round(t2, 4),
        )
        val diff =
            t
                .sub(t2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(diff < 1e-5f)
    }

    @Test
    fun softmaxNumericalStability() {
        val dev = Device.Cpu
        val xs = Tensor.new(floatArrayOf(1234.0f, 0.0f), Shape(2), dev)
        val sm = softmax(xs, 0)
        assertEquals(listOf(1.0f, 0.0f), sm.toVec1<Float>())
    }

    @Test
    fun ropeiCpu() {
        val device = Device.Cpu
        val bSize = 2
        val numHead = 5
        val seqLen = 10
        val headDim = 16
        val elCount = bSize * numHead * seqLen * headDim
        val rng = Random(299792458L)
        val src =
            Tensor.fromVec(
                List(elCount) { rng.nextFloat() },
                Shape(bSize, numHead, seqLen, headDim),
                device,
            )
        val cos =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val sin =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val rope1 = ropeI(src, cos, sin)
        val rope2 = ropeISlow(src, cos, sin)
        val sumDiff =
            rope1
                .sub(rope2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(sumDiff < 1e-4f)
    }

    @Test
    fun ropeCpu() {
        val device = Device.Cpu
        val bSize = 2
        val numHead = 5
        val seqLen = 10
        val headDim = 16
        val elCount = bSize * numHead * seqLen * headDim
        val rng = Random(299792458L)
        val src =
            Tensor.fromVec(
                List(elCount) { rng.nextFloat() },
                Shape(bSize, numHead, seqLen, headDim),
                device,
            )
        val cos =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val sin =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val rope1 = rope(src, cos, sin)
        val rope2 = ropeSlow(src, cos, sin)
        val sumDiff =
            rope1
                .sub(rope2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(sumDiff < 1e-4f)
    }

    @Test
    fun ropeThdCpu() {
        val device = Device.Cpu
        val bSize = 2
        val numHead = 5
        val seqLen = 10
        val headDim = 16
        val elCount = bSize * numHead * seqLen * headDim
        val rng = Random(299792458L)
        val src =
            Tensor.fromVec(
                List(elCount) { rng.nextFloat() },
                Shape(bSize, numHead, seqLen, headDim),
                device,
            )
        val cos =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val sin =
            Tensor.fromVec(
                List(seqLen * headDim / 2) { rng.nextFloat() },
                Shape(seqLen, headDim / 2),
                device,
            )
        val rope1 = ropeThd(src.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        val rope2 = ropeSlow(src, cos, sin)
        val sumDiff =
            rope1
                .sub(rope2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(sumDiff < 1e-4f)
    }

    @Test
    fun sigmoidCpu() {
        val device = Device.Cpu
        val data =
            floatArrayOf(
                3.0f,
                1.0f,
                4.0f,
                1.0f,
                5.0f,
                9.0f,
                2.0f,
                1.0f,
                7.0f,
                8.0f,
                2.0f,
                8.0f,
            )
        val tensor = Tensor.new(data, Shape(2, 2, 3), device)
        val s1 = sigmoid(tensor)
        val s2 = (tensor.neg().exp() + 1.0).recip()
        val diff =
            s1
                .sub(s2)
                .abs()
                .sumAll()
                .toVec0<Float>()
        assertTrue(diff < 1e-5f)
    }
}
