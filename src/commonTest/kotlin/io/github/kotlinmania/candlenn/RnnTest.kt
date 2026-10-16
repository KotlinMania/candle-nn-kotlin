// port-lint: tests tests/rnn.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toVec2Round
import kotlin.test.Test
import kotlin.test.assertEquals

class RnnTest {
    @Test
    fun lstm() {
        val cpu = Device.Cpu
        val wIh = Tensor.arange(0.0f, 24.0f, cpu).reshape(12, 2).cos()
        val wHh = Tensor.arange(0.0f, 36.0f, cpu).reshape(12, 3).sin()
        val bIh = Tensor.new(
            floatArrayOf(-1.0f, 1.0f, -0.5f, 2.0f, -1.0f, 1.0f, -0.5f, 2.0f, -1.0f, 1.0f, -0.5f, 2.0f),
            Shape(12),
            cpu,
        )
        val bHh = bIh.cos()
        val tensors = mapOf(
            "weight_ih_l0" to wIh,
            "weight_hh_l0" to wHh,
            "bias_ih_l0" to bIh,
            "bias_hh_l0" to bHh,
        )
        val vb = VarBuilder.fromTensors(tensors, DType.F32, cpu)
        val lstmLayer = lstm(2, 3, LSTMConfig.default(), vb)
        var state = lstmLayer.zeroState(1)
        for (inpVal in floatArrayOf(3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f)) {
            val inp = Tensor.new(floatArrayOf(inpVal, inpVal * 0.5f), Shape(1, 2), cpu)
            state = lstmLayer.step(inp, state)
        }
        assertEquals(listOf(listOf(0.9919f, 0.1738f, -0.1451f)), toVec2Round(state.h(), 4))
        assertEquals(listOf(listOf(5.725f, 0.4458f, -0.2908f)), toVec2Round(state.c(), 4))
    }

    @Test
    fun gru() {
        val cpu = Device.Cpu
        val wIh = Tensor.arange(0.0f, 18.0f, cpu).reshape(9, 2).cos()
        val wHh = Tensor.arange(0.0f, 27.0f, cpu).reshape(9, 3).sin()
        val bIh = Tensor.new(floatArrayOf(-1.0f, 1.0f, -0.5f, 2.0f, -1.0f, 1.0f, -0.5f, 2.0f, -1.0f), Shape(9), cpu)
        val bHh = bIh.cos()
        val tensors = mapOf(
            "weight_ih_l0" to wIh,
            "weight_hh_l0" to wHh,
            "bias_ih_l0" to bIh,
            "bias_hh_l0" to bHh,
        )
        val vb = VarBuilder.fromTensors(tensors, DType.F32, cpu)
        val gruLayer = gru(2, 3, GRUConfig.default(), vb)
        var state = gruLayer.zeroState(1)
        for (inpVal in floatArrayOf(3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f)) {
            val inp = Tensor.new(floatArrayOf(inpVal, inpVal * 0.5f), Shape(1, 2), cpu)
            state = gruLayer.step(inp, state)
        }
        assertEquals(listOf(listOf(0.0579f, 0.8836f, -0.9991f)), toVec2Round(state.h(), 4))
    }
}
