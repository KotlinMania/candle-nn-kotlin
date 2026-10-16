// port-lint: source optim.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.GradStore
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import io.github.kotlinmania.candle.backward
import kotlin.math.pow

/**
 * The interface optimizers should implement.
 */
public interface Optimizer {
    public fun step(grads: GradStore)

    public fun learningRate(): Double

    public fun setLearningRate(lr: Double)

    public fun backwardStep(loss: Tensor) {
        val grads = loss.backward()
        step(grads)
    }
}

/**
 * Optimizer for Stochastic Gradient Descent.
 *
 * Contrary to the PyTorch implementation of SGD, this version does not support momentum.
 */
public class SGD internal constructor(
    private val vars: MutableList<Var>,
    private var learningRate: Double,
) : Optimizer {
    public companion object {
        public fun new(vars: List<Var>, learningRate: Double): SGD {
            val filtered = vars.filter { it.asTensor().dtype().isFloat() }.toMutableList()
            return SGD(filtered, learningRate)
        }

        public fun empty(learningRate: Double): SGD =
            new(emptyList(), learningRate)

        public fun fromSlice(vars: List<Var>, learningRate: Double): SGD =
            new(vars.toList(), learningRate)
    }

    override fun learningRate(): Double = learningRate

    override fun setLearningRate(lr: Double) {
        learningRate = lr
    }

    override fun step(grads: GradStore) {
        for (v in vars) {
            val grad = grads.get(v.asTensor())
            if (grad != null) {
                v.set(v.asTensor().sub(grad * learningRate))
            }
        }
    }

    public fun intoInner(): List<Var> = vars

    public fun push(v: Var) {
        vars.add(v)
    }
}

public data class ParamsAdamW(
    public val lr: Double = 0.001,
    public val beta1: Double = 0.9,
    public val beta2: Double = 0.999,
    public val eps: Double = 1e-8,
    public val weightDecay: Double = 0.01,
) {
    public companion object {
        public fun default(): ParamsAdamW = ParamsAdamW()
    }
}

internal class VarAdamW(
    val variable: Var,
    val firstMoment: Var,
    val secondMoment: Var,
)

public class AdamW internal constructor(
    private val vars: List<VarAdamW>,
    private var stepT: Int,
    private var params: ParamsAdamW,
) : Optimizer {
    public companion object {
        public fun new(vars: List<Var>, params: ParamsAdamW): AdamW {
            val adamVars = vars
                .filter { it.asTensor().dtype().isFloat() }
                .map { v ->
                    val dtype = v.asTensor().dtype()
                    val shape = v.asTensor().shape()
                    val device = v.asTensor().device()
                    val firstMoment = Var.zeros(shape, dtype, device)
                    val secondMoment = Var.zeros(shape, dtype, device)
                    VarAdamW(
                        variable = v,
                        firstMoment = firstMoment,
                        secondMoment = secondMoment,
                    )
                }
            return AdamW(
                vars = adamVars,
                stepT = 0,
                params = params,
            )
        }

        public fun newLr(vars: List<Var>, learningRate: Double): AdamW {
            val params = ParamsAdamW(lr = learningRate)
            return new(vars, params)
        }

        public fun empty(params: ParamsAdamW): AdamW =
            new(emptyList(), params)

        public fun fromSlice(vars: List<Var>, params: ParamsAdamW): AdamW =
            new(vars.toList(), params)
    }

    override fun learningRate(): Double = params.lr

    override fun setLearningRate(lr: Double) {
        params = params.copy(lr = lr)
    }

    public fun params(): ParamsAdamW = params

    public fun setParams(newParams: ParamsAdamW) {
        params = newParams
    }

    override fun step(grads: GradStore) {
        stepT += 1
        val lr = params.lr
        val lambda = params.weightDecay
        val lrLambda = lr * lambda
        val beta1 = params.beta1
        val beta2 = params.beta2
        val scaleM = 1.0 / (1.0 - beta1.pow(stepT))
        val scaleV = 1.0 / (1.0 - beta2.pow(stepT))
        for (v in vars) {
            val theta = v.variable
            val m = v.firstMoment
            val second = v.secondMoment
            val g = grads.get(theta.asTensor())
            if (g != null) {
                val nextM = (m.asTensor() * beta1) + (g * (1.0 - beta1))
                val nextV = (second.asTensor() * beta2) + (g.sqr() * (1.0 - beta2))
                val mHat = nextM * scaleM
                val vHat = nextV * scaleV
                val nextTheta0 = theta.asTensor() * (1.0 - lrLambda)
                val adjustedGrad = mHat / (vHat.sqrt() + params.eps)
                val nextTheta = nextTheta0 - (adjustedGrad * lr)
                m.set(nextM)
                second.set(nextV)
                theta.set(nextTheta)
            }
        }
    }
}
