// port-lint: tests tests/layer_norm.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.toVec3Round
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class LayerNormTest {
    @Test
    fun layerNorm() {
        val device = Device.Cpu
        val w = Tensor.new(floatArrayOf(3.0f), Shape(1), device)
        val b = Tensor.new(floatArrayOf(0.5f), Shape(1), device)
        val ln2 = LayerNorm.new(Tensor.cat(listOf(w, w), 0), Tensor.cat(listOf(b, b), 0), 1e-8)
        val ln3 =
            LayerNorm.new(
                Tensor.cat(listOf(w, w, w), 0),
                Tensor.cat(listOf(b, b, b), 0),
                1e-8,
            )
        val ln = LayerNorm.new(w, b, 1e-8)
        assertEquals(1e-8, ln.eps())
        assertTrue(ln.removeMean())

        val two = Tensor.new(floatArrayOf(2.0f), Shape(1, 1, 1), device)
        val res1 = ln.forward(two).flattenAll()
        assertEquals(listOf(0.5f), res1.toVec1<Float>())

        val inp2 = Tensor.new(floatArrayOf(4.0f, 0.0f), Shape(1, 1, 2), device)
        val res2 = ln2.forward(inp2)
        assertEquals(listOf(listOf(listOf(3.5f, -2.5f))), res2.toVec3<Float>())

        val inp3 =
            Tensor.new(
                floatArrayOf(
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                    9.0f,
                    8.0f,
                    7.0f,
                ),
                Shape(1, 3, 3),
                device,
            )
        val res3 = ln3.forward(inp3)
        assertEquals(
            listOf(
                listOf(
                    listOf(-3.1742f, 0.5f, 4.1742f),
                    listOf(-3.1742f, 0.5f, 4.1742f),
                    listOf(4.1742f, 0.5f, -3.1742f),
                ),
            ),
            toVec3Round(res3, 4),
        )
        val mean = res3.sumKeepdim(2).div(3.0)
        assertEquals(
            listOf(listOf(listOf(0.5f), listOf(0.5f), listOf(0.5f))),
            toVec3Round(mean, 4),
        )
        val std =
            res3
                .broadcastSub(mean)
                .sqr()
                .sumKeepdim(2)
                .sqrt()
                .div(3.0)
        assertEquals(
            listOf(listOf(listOf(1.7321f), listOf(1.7321f), listOf(1.7321f))),
            toVec3Round(std, 4),
        )

        val rms = LayerNorm.rmsNorm(Tensor.new(floatArrayOf(1.0f), Shape(1), device), 1e-5)
        assertEquals(1e-5, rms.eps())
        assertFalse(rms.removeMean())
    }

    @Test
    fun rmsNormFromVarmap() {
        val device = Device.Cpu
        val varmap = VarMap.new()
        val vb = VarBuilder.fromVarmap(varmap, DType.F32, device)
        val rms = rmsNorm(8, 1e-6, vb.pp("norm"))
        assertEquals(listOf(8), rms.weight().dims())
        assertTrue(varmap.data().containsKey("norm.weight"))
        assertFalse(varmap.data().containsKey("norm.bias"))
    }

    @Test
    fun layerNormFromVarmap() {
        val device = Device.Cpu
        val varmap = VarMap.new()
        val vb = VarBuilder.fromVarmap(varmap, DType.F32, device)
        val ln = layerNorm(4, LayerNormConfig.fromEps(1e-5), vb.pp("ln"))
        assertEquals(listOf(4), ln.weight().dims())
        assertNotNull(ln.bias())
        assertEquals(listOf(4), ln.bias()!!.dims())
        val data = varmap.data()
        assertTrue(data.containsKey("ln.weight"))
        assertTrue(data.containsKey("ln.bias"))
    }

    @Test
    fun layerNormGammaBetaAliases() {
        val device = Device.Cpu
        val gamma = Tensor.new(floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f), Shape(4), device)
        val beta = Tensor.new(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f), Shape(4), device)
        val tensors =
            mapOf(
                "ln.gamma" to gamma,
                "ln.beta" to beta,
            )
        val vb = VarBuilder.fromTensors(tensors, DType.F32, device)
        val ln = layerNorm(4, LayerNormConfig.fromEps(1e-5), vb.pp("ln"))
        assertEquals(listOf(4), ln.weight().dims())
        assertNotNull(ln.bias())
        assertEquals(listOf(4), ln.bias()!!.dims())
    }
}
