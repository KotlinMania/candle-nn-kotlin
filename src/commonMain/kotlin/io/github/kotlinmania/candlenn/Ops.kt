// port-lint: source ops.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CustomOp1
import io.github.kotlinmania.candle.CustomOp2
import io.github.kotlinmania.candle.CustomOp3
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Map1
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.ModuleT
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.applyOp1
import io.github.kotlinmania.candle.applyOp1NoBwd
import io.github.kotlinmania.candle.applyOp2NoBwd
import io.github.kotlinmania.candle.applyOp3NoBwd
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.toIndex
import io.github.kotlinmania.candle.unaryMap
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Applies the softmax function to the input tensor, rescaling the element so that elements on
 * a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
 */
public fun softmax(
    xs: Tensor,
    dim: Int,
): Tensor {
    val dimIdx = Shape.toIndex(dim, xs.shape(), "softmax")
    val max = xs.maxKeepdim(dimIdx)
    val diff = xs.broadcastSub(max)
    val num = diff.exp()
    val sum = num.sumKeepdim(dimIdx)
    return num.broadcastDiv(sum)
}

/**
 * Applies the softmax function to the input tensor along the given dimension.
 */
public fun softmax(
    xs: Tensor,
    dim: io.github.kotlinmania.candle.Dim,
): Tensor = softmax(xs, dim.toIndex(xs.shape(), "softmax"))

public fun logSoftmax(
    xs: Tensor,
    dim: Int,
): Tensor {
    val d = Shape.toIndex(dim, xs.shape(), "log-softmax")
    val max = xs.maxKeepdim(d)
    val diff = xs.broadcastSub(max)
    val sumExp = diff.exp().sumKeepdim(d)
    val logSm = diff.broadcastSub(sumExp.log())
    return logSm
}

public fun logSoftmax(
    xs: Tensor,
    dim: io.github.kotlinmania.candle.Dim,
): Tensor = logSoftmax(xs, dim.toIndex(xs.shape(), "log-softmax"))

public fun silu(xs: Tensor): Tensor = xs.silu()

public fun swiglu(xs: Tensor): Tensor {
    val chunks = xs.chunk(2, D.Minus1)
    return chunks[0].silu() * chunks[1]
}

internal class Sigmoid :
    CustomOp1,
    Map1 {
    override fun name(): String = "sigmoid"

    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> = unaryMap(vs, layout) { v -> 1.0 / (1.0 + exp(-v)) }

    override fun cpuFwd(
        storage: CpuStorage,
        layout: Layout,
    ): Pair<CpuStorage, Shape> = Pair(map(storage, layout), layout.shape())

    override fun bwd(
        arg: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Tensor {
        if (arg.dtype() != res.dtype()) {
            bail("sigmoid bwd dtype mismatch")
        }
        val dLocSigmoid = res.onesLike().sub(res).mul(res)
        return gradRes.mul(dLocSigmoid)
    }
}

internal fun Tensor.sumKeepdim(dim: io.github.kotlinmania.candle.Dim): Tensor = sumKeepdim(dim.toIndex(shape(), "sum-keepdim"))

internal operator fun Tensor.plus(rhs: Double): Tensor = add(rhs)

internal operator fun Tensor.minus(rhs: Double): Tensor = sub(rhs)

internal operator fun Tensor.times(rhs: Double): Tensor = mul(rhs)

internal operator fun Double.times(rhs: Tensor): Tensor = rhs.mul(this)

internal operator fun Double.plus(rhs: Tensor): Tensor = rhs.add(this)

internal operator fun Double.minus(rhs: Tensor): Tensor = rhs.affine(-1.0, this)

public fun sigmoid(xs: Tensor): Tensor = xs.applyOp1(Sigmoid())

public fun hardSigmoid(xs: Tensor): Tensor = ((xs + 3.0) / 6.0).clamp(0.0, 1.0)

public fun LeakyRelu(
    xs: Tensor,
    negativeSlope: Double,
): Tensor {
    val zeros = xs.zerosLike()
    return xs.maximum(zeros) + xs.minimum(zeros) * negativeSlope
}

public fun leakyRelu(
    xs: Tensor,
    negativeSlope: Double,
): Tensor = LeakyRelu(xs, negativeSlope)

public fun selu(
    xs: Tensor,
    alpha: Float,
    gamma: Float,
): Tensor {
    val isPos = xs.gt(xs.zerosLike())
    val t = xs.exp().affine(alpha.toDouble(), -alpha.toDouble())
    val res = isPos.whereCond(xs, t)
    return res.affine(gamma.toDouble(), 0.0)
}

public fun dropout(
    xs: Tensor,
    dropP: Float,
): Tensor {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be tobe able to generate boolean mask Using just a bit of
    // entropy per element rather than generating a full float per element.
    if (dropP < 0.0f || dropP >= 1.0f) {
        bail("dropout probability has to be in [0, 1), got $dropP")
    }
    val rand = Tensor.rand(0.0f, 1.0f, xs.shape(), xs.device())
    val scale = 1.0 / (1.0 - dropP.toDouble())
    val dropPVal = Tensor.new(floatArrayOf(dropP), Shape(emptyList()), xs.device()).broadcastAs(xs.shape())
    val mask = (rand.ge(dropPVal).toDtype(xs.dtype()) * scale)
    return xs * mask
}

public class Dropout(
    private val dropP: Float,
) : ModuleT {
    public companion object {
        public fun new(dropP: Float): Dropout = Dropout(dropP)
    }

    public fun forward(
        xs: Tensor,
        train: Boolean,
    ): Tensor = if (train) dropout(xs, dropP) else xs

    override fun forwardT(
        xs: Tensor,
        train: Boolean,
    ): Tensor = forward(xs, train)
}

internal class SoftmaxLastDim :
    CustomOp1,
    Map1 {
    override fun name(): String = "softmax-last-dim"

    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        if (!layout.isContiguous()) {
            throw CandleException.RequiresContiguous("softmax-last-dim")
        }
        val elCount = layout.shape().elemCount()
        if (elCount == 0) return emptyList()
        val srcIndex = layout.startOffset()
        val dims = layout.dims()
        val dimM1 = dims[dims.size - 1]
        val dst = MutableList(elCount) { 0.0 }
        var srcStart = srcIndex
        var dstStart = 0
        while (dstStart < elCount) {
            var max = Double.NEGATIVE_INFINITY
            for (i in 0 until dimM1) {
                val v = vs[srcStart + i]
                if (v > max) max = v
            }
            var sumExp = 0.0
            for (i in 0 until dimM1) {
                val e = exp(vs[srcStart + i] - max)
                dst[dstStart + i] = e
                sumExp += e
            }
            for (i in 0 until dimM1) {
                dst[dstStart + i] /= sumExp
            }
            srcStart += dimM1
            dstStart += dimM1
        }
        return dst
    }

    override fun cpuFwd(
        storage: CpuStorage,
        layout: Layout,
    ): Pair<CpuStorage, Shape> = Pair(map(storage, layout), layout.shape())
}

public fun softmaxLastDim(xs: Tensor): Tensor = xs.applyOp1NoBwd(SoftmaxLastDim())

internal class RmsNormOp(
    private val eps: Float,
) : CustomOp2 {
    override fun name(): String = "rms-norm"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
    ): Pair<CpuStorage, Shape> {
        if (!l1.isContiguous() || !l2.isContiguous()) {
            throw CandleException.RequiresContiguous("rms-norm")
        }
        if (s1.dtype() != s2.dtype()) {
            throw CandleException.DTypeMismatchBinaryOp(s1.dtype(), s2.dtype(), "rms-norm")
        }
        val elCount = l1.shape().elemCount()
        if (elCount == 0) return Pair(s1, l1.shape())
        val dims = l1.dims()
        val dimM1 = dims[dims.size - 1]
        val v1 = s1.asSlice()
        val v2 = s2.asSlice()
        val srcIndex = l1.startOffset()
        val alphaIndex = l2.startOffset()
        val dst = MutableList(elCount) { 0.0 }
        var srcStart = srcIndex
        var dstStart = 0
        while (dstStart < elCount) {
            var sum2 = 0.0
            for (i in 0 until dimM1) {
                val v = v1[srcStart + i]
                sum2 += v * v
            }
            val m = sum2 / dimM1.toDouble()
            val s = sqrt(m + eps.toDouble())
            for (i in 0 until dimM1) {
                dst[dstStart + i] = v1[srcStart + i] / s * v2[alphaIndex + i]
            }
            srcStart += dimM1
            dstStart += dimM1
        }
        val outStorage = wrapCpuStorage(s1.dtype(), dst)
        return Pair(outStorage, l1.shape())
    }
}

private fun wrapCpuStorage(
    dtype: DType,
    dst: List<Double>,
): CpuStorage =
    when (dtype) {
        DType.U8 -> {
            CpuStorage.U8(dst)
        }

        DType.U32 -> {
            CpuStorage.U32(dst)
        }

        DType.I16 -> {
            CpuStorage.I16(dst)
        }

        DType.I32 -> {
            CpuStorage.I32(dst)
        }

        DType.I64 -> {
            CpuStorage.I64(dst)
        }

        DType.BF16 -> {
            CpuStorage.BF16(dst)
        }

        DType.F16 -> {
            CpuStorage.F16(dst)
        }

        DType.F32 -> {
            CpuStorage.F32(dst)
        }

        DType.F64 -> {
            CpuStorage.F64(dst)
        }

        DType.F8E4M3 -> {
            CpuStorage.F8E4M3(dst)
        }

        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            throw CandleException.UnsupportedDTypeForOp(dtype, "custom-op")
        }
    }

public fun rmsNormSlow(
    x: Tensor,
    alpha: Tensor,
    eps: Float,
): Tensor {
    val xDtype = x.dtype()
    val internalDtype =
        when (xDtype) {
            DType.F16, DType.BF16 -> DType.F32
            else -> xDtype
        }
    val hiddenSize = x.dim(D.Minus1)
    val xCast = x.toDtype(internalDtype)
    val normX = (xCast.sqr().sumKeepdim(D.Minus1) / hiddenSize.toDouble())
    val xNormed = xCast.broadcastDiv((normX + eps.toDouble()).sqrt())
    return xNormed.toDtype(xDtype).broadcastMul(alpha)
}

public fun rmsNorm(
    xs: Tensor,
    alpha: Tensor,
    eps: Float,
): Tensor {
    val hiddenSizeXs = xs.dim(D.Minus1)
    val hiddenSizeAlpha = alpha.dims1()
    if (hiddenSizeXs != hiddenSizeAlpha) {
        bail("shape mismatch in rms-norm ${xs.shape()} ${alpha.shape()}")
    }
    return xs.applyOp2NoBwd(alpha, RmsNormOp(eps))
}

internal class LayerNormOp(
    private val eps: Float,
) : CustomOp3 {
    override fun name(): String = "layer-norm"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape> {
        if (!l1.isContiguous() || !l2.isContiguous() || !l3.isContiguous()) {
            throw CandleException.RequiresContiguous("layer-norm")
        }
        if (s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype()) {
            throw CandleException.DTypeMismatchBinaryOp(s1.dtype(), s2.dtype(), "layer-norm")
        }
        val elCount = l1.shape().elemCount()
        if (elCount == 0) return Pair(s1, l1.shape())
        val dims = l1.dims()
        val dimM1 = dims[dims.size - 1]
        val v1 = s1.asSlice()
        val v2 = s2.asSlice()
        val v3 = s3.asSlice()
        val srcIndex = l1.startOffset()
        val alphaIndex = l2.startOffset()
        val betaIndex = l3.startOffset()
        val dst = MutableList(elCount) { 0.0 }
        var srcStart = srcIndex
        var dstStart = 0
        while (dstStart < elCount) {
            var sum = 0.0
            var sum2 = 0.0
            for (i in 0 until dimM1) {
                val v = v1[srcStart + i]
                sum += v
                sum2 += v * v
            }
            val mean = sum / dimM1.toDouble()
            val variance = sum2 / dimM1.toDouble() - mean * mean
            val s = sqrt(variance + eps.toDouble())
            for (i in 0 until dimM1) {
                dst[dstStart + i] = (v1[srcStart + i] - mean) / s * v2[alphaIndex + i] + v3[betaIndex + i]
            }
            srcStart += dimM1
            dstStart += dimM1
        }
        val outStorage = wrapCpuStorage(s1.dtype(), dst)
        return Pair(outStorage, l1.shape())
    }
}

public fun layerNormSlow(
    x: Tensor,
    alpha: Tensor,
    beta: Tensor,
    eps: Float,
): Tensor {
    val xDtype = x.dtype()
    val internalDtype =
        when (xDtype) {
            DType.F16, DType.BF16 -> DType.F32
            else -> xDtype
        }
    val hiddenSize = x.dim(D.Minus1)
    val xCast = x.toDtype(internalDtype)
    val xSubMean =
        run {
            val meanX = (xCast.sumKeepdim(D.Minus1) / hiddenSize.toDouble())
            xCast.broadcastSub(meanX)
        }
    val normX = (xSubMean.sqr().sumKeepdim(D.Minus1) / hiddenSize.toDouble())
    val xNormed = xSubMean.broadcastDiv((normX + eps.toDouble()).sqrt())
    return xNormed.toDtype(xDtype).broadcastMul(alpha).broadcastAdd(beta)
}

public fun layerNorm(
    xs: Tensor,
    alpha: Tensor,
    beta: Tensor,
    eps: Float,
): Tensor {
    val hiddenSizeXs = xs.dim(D.Minus1)
    val hiddenSizeAlpha = alpha.dims1()
    val hiddenSizeBeta = beta.dims1()
    if (hiddenSizeXs != hiddenSizeAlpha || hiddenSizeXs != hiddenSizeBeta) {
        bail("shape mismatch in layer-norm ${xs.shape()} ${alpha.shape()} ${beta.shape()}")
    }
    return xs.applyOp3NoBwd(alpha, beta, LayerNormOp(eps))
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
public fun pixelShuffle(
    xs: Tensor,
    upscaleFactor: Int,
): Tensor {
    val dims = xs.dims4()
    val bSize = dims.d0
    val c = dims.d1
    val h = dims.d2
    val w = dims.d3
    val outC = c / upscaleFactor / upscaleFactor
    return xs
        .reshape(Shape(bSize, outC, upscaleFactor, upscaleFactor, h, w))
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(Shape(bSize, outC, h * upscaleFactor, w * upscaleFactor))
}

public fun pixelUnshuffle(
    xs: Tensor,
    downscaleFactor: Int,
): Tensor {
    val dims = xs.dims4()
    val bSize = dims.d0
    val c = dims.d1
    val h = dims.d2
    val w = dims.d3
    val outC = c * downscaleFactor * downscaleFactor
    return xs
        .reshape(Shape(bSize, c, h / downscaleFactor, downscaleFactor, w / downscaleFactor, downscaleFactor))
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(Shape(bSize, outC, h / downscaleFactor, w / downscaleFactor))
}

public fun replicationPad2d(
    xs: Tensor,
    pad: Int,
): Tensor {
    when (pad) {
        0 -> {
            return xs
        }

        1 -> {
            val dims = xs.dims4()
            val h = dims.d2
            val w = dims.d3
            val first = xs.narrow(3, 0, 1)
            val last = xs.narrow(3, w - 1, 1)
            val xsPaddedW = Tensor.cat(listOf(first, xs, last), 3)
            val firstH = xsPaddedW.narrow(2, 0, 1)
            val lastH = xsPaddedW.narrow(2, h - 1, 1)
            return Tensor.cat(listOf(firstH, xsPaddedW, lastH), 2)
        }

        else -> {
            bail("replication-pad with size $pad is not supported")
        }
    }
}

public class Identity : Module {
    public companion object {
        public fun new(): Identity = Identity()
    }

    override fun forward(xs: Tensor): Tensor = xs
}

public fun sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: Float,
    softcapping: Float,
): Tensor {
    val att = (q.matmul(k.t()) * scale.toDouble())
    val attCapped =
        if (softcapping == 1.0f) {
            att
        } else {
            (att / softcapping.toDouble()).tanh() * softcapping.toDouble()
        }
    val attSoftmax = softmaxLastDim(attCapped)
    return attSoftmax.matmul(v)
}

/**
 * Applies the Mish activation function: `x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`.
 */
public fun mish(xs: Tensor): Tensor = xs * (1.0 + xs.exp()).log().tanh()
