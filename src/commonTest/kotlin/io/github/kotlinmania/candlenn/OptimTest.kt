// port-lint: tests tests/optim.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import io.github.kotlinmania.candle.toVec0Round
import io.github.kotlinmania.candle.toVec2Round
import kotlin.test.Test
import kotlin.test.assertEquals

class OptimTest {
    @Test
    fun sgdOptim() {
        val x = Var.fromTensor(Tensor.new(floatArrayOf(0.0f), Shape.SCALAR, Device.Cpu))
        val sgd = SGD.new(listOf(x), 0.1)
        val xt = x.asTensor()
        repeat(100) {
            val loss = xt.sub(4.2).mul(xt.sub(4.2))
            sgd.backwardStep(loss)
        }
        assertEquals(4.199999f, x.asTensor().toScalar<Float>(), 1e-6f)
    }

    @Test
    fun sgdLinearRegression() {
        val wGen = Tensor.new(floatArrayOf(3.0f, 1.0f), Shape(1, 2), Device.Cpu)
        val bGen = Tensor.new(floatArrayOf(-2.0f), Shape.SCALAR, Device.Cpu)
        val gen = Linear.new(wGen, bGen)
        val sampleXs =
            Tensor.new(
                floatArrayOf(
                    2.0f,
                    1.0f,
                    7.0f,
                    4.0f,
                    -4.0f,
                    12.0f,
                    5.0f,
                    8.0f,
                ),
                Shape(4, 2),
                Device.Cpu,
            )
        val sampleYs = gen.forward(sampleXs)

        val w = Var.fromTensor(Tensor.new(floatArrayOf(0.0f, 0.0f), Shape(1, 2), Device.Cpu))
        val b = Var.fromTensor(Tensor.new(floatArrayOf(0.0f), Shape.SCALAR, Device.Cpu))
        val sgd = SGD.new(listOf(w, b), 0.004)
        val lin = Linear.new(w.asTensor(), b.asTensor())
        repeat(1000) {
            val ys = lin.forward(sampleXs)
            val loss = ys.sub(sampleYs).sqr().sumAll()
            sgd.backwardStep(loss)
        }
        assertEquals(listOf(listOf(2.9983f, 0.9979f)), toVec2Round(w.asTensor(), 4))
        assertEquals(-1.9797f, toVec0Round(b.asTensor(), 4))
    }

    @Test
    fun adamwLinearRegression() {
        val wGen = Tensor.new(floatArrayOf(3.0f, 1.0f), Shape(1, 2), Device.Cpu)
        val bGen = Tensor.new(floatArrayOf(-2.0f), Shape.SCALAR, Device.Cpu)
        val gen = Linear.new(wGen, bGen)
        val sampleXs =
            Tensor.new(
                floatArrayOf(
                    2.0f,
                    1.0f,
                    7.0f,
                    4.0f,
                    -4.0f,
                    12.0f,
                    5.0f,
                    8.0f,
                ),
                Shape(4, 2),
                Device.Cpu,
            )
        val sampleYs = gen.forward(sampleXs)

        val w = Var.fromTensor(Tensor.new(floatArrayOf(0.0f, 0.0f), Shape(1, 2), Device.Cpu))
        val b = Var.fromTensor(Tensor.new(floatArrayOf(0.0f), Shape.SCALAR, Device.Cpu))
        val params = ParamsAdamW(lr = 0.1)
        val opt = AdamW.new(listOf(w, b), params)
        val lin = Linear.new(w.asTensor(), b.asTensor())
        repeat(100) {
            val ys = lin.forward(sampleXs)
            val loss = ys.sub(sampleYs).sqr().sumAll()
            opt.backwardStep(loss)
        }
        assertEquals(listOf(listOf(2.7257f, 0.7097f)), toVec2Round(w.asTensor(), 4))
        assertEquals(0.7873f, toVec0Round(b.asTensor(), 4))
    }

    @Test
    fun adamwLinearRegressionVarmap() {
        val wGen = Tensor.new(floatArrayOf(3.0f, 1.0f), Shape(1, 2), Device.Cpu)
        val bGen = Tensor.new(floatArrayOf(-2.0f), Shape.SCALAR, Device.Cpu)
        val gen = Linear.new(wGen, bGen)
        val sampleXs =
            Tensor.new(
                floatArrayOf(
                    2.0f,
                    1.0f,
                    7.0f,
                    4.0f,
                    -4.0f,
                    12.0f,
                    5.0f,
                    8.0f,
                ),
                Shape(4, 2),
                Device.Cpu,
            )
        val sampleYs = gen.forward(sampleXs)

        val varMap = VarMap.new()
        val w = varMap.get(Pair(1, 2), "w", Init.Const(0.0), DType.F32, Device.Cpu)
        val b = varMap.get(Shape.SCALAR, "b", Init.Const(0.0), DType.F32, Device.Cpu)
        val params = ParamsAdamW(lr = 0.1)
        val opt = AdamW.new(varMap.allVars(), params)
        val lin = Linear.new(w, b)
        repeat(100) {
            val ys = lin.forward(sampleXs)
            val loss = ys.sub(sampleYs).sqr().sumAll()
            opt.backwardStep(loss)
        }
        assertEquals(listOf(listOf(2.7257f, 0.7097f)), toVec2Round(lin.weight(), 4))
        assertEquals(0.7873f, toVec0Round(lin.bias()!!, 4))

        varMap.setOne("w", Tensor.zeros(Shape(1, 2), DType.F32, Device.Cpu))
        varMap.setOne("b", Tensor.ones(Shape.SCALAR, DType.F32, Device.Cpu))

        assertEquals(listOf(listOf(0.0f, 0.0f)), toVec2Round(lin.weight(), 4))
        assertEquals(1.0f, toVec0Round(lin.bias()!!, 4))
    }
}
