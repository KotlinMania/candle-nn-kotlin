// port-lint: tests tests/loss.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toVec0Round
import kotlin.test.Test
import kotlin.test.assertEquals

class LossTest {
    @Test
    fun nllAndCrossEntropy() {
        val cpu = Device.Cpu
        val input =
            Tensor.new(
                floatArrayOf(
                    1.1050f,
                    0.3013f,
                    -1.5394f,
                    -2.1528f,
                    -0.8634f,
                    1.0730f,
                    -0.9419f,
                    -0.1670f,
                    -0.6582f,
                    0.5061f,
                    0.8318f,
                    1.1154f,
                    -0.3610f,
                    0.5351f,
                    1.0830f,
                ),
                Shape(3, 5),
                cpu,
            )
        val target = Tensor.new(uintArrayOf(1u, 0u, 4u), Shape(3), cpu)

        val logSm = logSoftmax(input, 1)
        val loss1 = nll(logSm, target)
        assertEquals(1.1312f, toVec0Round(loss1, 4))
        val loss2 = crossEntropy(input, target)
        assertEquals(1.1312f, toVec0Round(loss2, 4))
    }

    @Test
    fun binaryCrossEntropyWithLogitTest() {
        val cpu = Device.Cpu
        val inp =
            Tensor.new(
                floatArrayOf(
                    2.3611f,
                    -0.8813f,
                    -0.5006f,
                    -0.2178f,
                    0.0419f,
                    0.0763f,
                    -1.0457f,
                    -1.6692f,
                    -1.0494f,
                    0.8111f,
                    1.5723f,
                    1.2315f,
                    1.3081f,
                    0.6641f,
                    1.1802f,
                    -0.2547f,
                    0.5292f,
                    0.7636f,
                    0.3692f,
                    -0.8318f,
                ),
                Shape(5, 4),
                cpu,
            )
        val target =
            Tensor.new(
                floatArrayOf(
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                ),
                Shape(5, 4),
                cpu,
            )
        val loss = binaryCrossEntropyWithLogit(inp, target)
        assertEquals(0.8224f, toVec0Round(loss, 4))
    }

    @Test
    fun huberLoss() {
        val cpu = Device.Cpu
        val inp =
            Tensor.new(
                floatArrayOf(
                    2.3611f,
                    -0.8813f,
                    -0.5006f,
                    -0.2178f,
                    0.0419f,
                    0.0763f,
                    -1.0457f,
                    -1.6692f,
                    -1.0494f,
                    0.8111f,
                    1.5723f,
                    1.2315f,
                    1.3081f,
                    0.6641f,
                    1.1802f,
                    -0.2547f,
                    0.5292f,
                    0.7636f,
                    0.3692f,
                    -0.8318f,
                ),
                Shape(5, 4),
                cpu,
            )
        val target =
            Tensor.new(
                floatArrayOf(
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    1.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f,
                    0.0f,
                ),
                Shape(5, 4),
                cpu,
            )
        val loss1 = huber(inp, target, 1.0)
        assertEquals(0.4734f, toVec0Round(loss1, 4))
        val loss2 = huber(inp, target, 0.88)
        assertEquals(0.4483f, toVec0Round(loss2, 4))
    }
}
