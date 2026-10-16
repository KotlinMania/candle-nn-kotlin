// port-lint: source rotary_emb.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CustomOp3
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.applyOp3NoBwd
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.dims4

private fun wrapRopeCpuStorage(dtype: DType, dst: List<Double>): CpuStorage =
    when (dtype) {
        DType.U8 -> CpuStorage.U8(dst)
        DType.U32 -> CpuStorage.U32(dst)
        DType.I16 -> CpuStorage.I16(dst)
        DType.I32 -> CpuStorage.I32(dst)
        DType.I64 -> CpuStorage.I64(dst)
        DType.BF16 -> CpuStorage.BF16(dst)
        DType.F16 -> CpuStorage.F16(dst)
        DType.F32 -> CpuStorage.F32(dst)
        DType.F64 -> CpuStorage.F64(dst)
        DType.F8E4M3 -> CpuStorage.F8E4M3(dst)
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
            throw CandleException.UnsupportedDTypeForOp(dtype, "rope")
    }

internal class RopeIOp : CustomOp3 {
    override fun name(): String = "rope-i"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape> {
        if (!l1.isContiguous() || !l2.isContiguous() || !l3.isContiguous()) {
            throw CandleException.RequiresContiguous("rope-i")
        }
        val dims = l1.shape().dims4()
        val b = dims.d0
        val h = dims.d1
        val t = dims.d2
        val d = dims.d3
        val elCount = b * h * t * d
        val src = s1.asSlice()
        val cos = s2.asSlice()
        val sin = s3.asSlice()
        val srcOffset = l1.startOffset()
        val cosOffset = l2.startOffset()
        val sinOffset = l3.startOffset()
        val cosDims = l2.dims()
        val isBatchedCos = cosDims.size == 3
        val halfD = d / 2
        val dst = MutableList(elCount) { 0.0 }

        for (bIdx in 0 until b) {
            val cosBOffset = if (isBatchedCos) bIdx * t * halfD else 0
            for (hIdx in 0 until h) {
                for (tIdx in 0 until t) {
                    val baseIdx = ((bIdx * h + hIdx) * t + tIdx) * d
                    val csBase = cosBOffset + tIdx * halfD
                    for (i in 0 until halfD) {
                        val x1 = src[srcOffset + baseIdx + 2 * i]
                        val x2 = src[srcOffset + baseIdx + 2 * i + 1]
                        val c = cos[cosOffset + csBase + i]
                        val s = sin[sinOffset + csBase + i]
                        dst[baseIdx + 2 * i] = x1 * c - x2 * s
                        dst[baseIdx + 2 * i + 1] = x1 * s + x2 * c
                    }
                }
            }
        }
        return Pair(wrapRopeCpuStorage(s1.dtype(), dst), l1.shape())
    }

    override fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        arg3: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Triple<Tensor?, Tensor?, Tensor?> {
        if (arg1.dtype() != res.dtype()) {
            bail("rope-i bwd dtype mismatch")
        }
        val gradX = ropeI(gradRes, arg2, arg3.neg())
        return Triple(gradX, null, null)
    }
}

internal class RopeOp : CustomOp3 {
    override fun name(): String = "rope"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape> {
        if (!l1.isContiguous() || !l2.isContiguous() || !l3.isContiguous()) {
            throw CandleException.RequiresContiguous("rope")
        }
        val dims = l1.shape().dims4()
        val b = dims.d0
        val h = dims.d1
        val t = dims.d2
        val d = dims.d3
        val elCount = b * h * t * d
        val src = s1.asSlice()
        val cos = s2.asSlice()
        val sin = s3.asSlice()
        val srcOffset = l1.startOffset()
        val cosOffset = l2.startOffset()
        val sinOffset = l3.startOffset()
        val cosDims = l2.dims()
        val isBatchedCos = cosDims.size == 3
        val halfD = d / 2
        val dst = MutableList(elCount) { 0.0 }

        for (bIdx in 0 until b) {
            val cosBOffset = if (isBatchedCos) bIdx * t * halfD else 0
            for (hIdx in 0 until h) {
                for (tIdx in 0 until t) {
                    val baseIdx = ((bIdx * h + hIdx) * t + tIdx) * d
                    val csBase = cosBOffset + tIdx * halfD
                    for (i in 0 until halfD) {
                        val x1 = src[srcOffset + baseIdx + i]
                        val x2 = src[srcOffset + baseIdx + halfD + i]
                        val c = cos[cosOffset + csBase + i]
                        val s = sin[sinOffset + csBase + i]
                        dst[baseIdx + i] = x1 * c - x2 * s
                        dst[baseIdx + halfD + i] = x1 * s + x2 * c
                    }
                }
            }
        }
        return Pair(wrapRopeCpuStorage(s1.dtype(), dst), l1.shape())
    }

    override fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        arg3: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Triple<Tensor?, Tensor?, Tensor?> {
        if (arg1.dtype() != res.dtype()) {
            bail("rope bwd dtype mismatch")
        }
        val gradX = rope(gradRes, arg2, arg3.neg())
        return Triple(gradX, null, null)
    }
}

internal class RopeThdOp : CustomOp3 {
    override fun name(): String = "rope-thd"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape> {
        if (!l1.isContiguous() || !l2.isContiguous() || !l3.isContiguous()) {
            throw CandleException.RequiresContiguous("rope-thd")
        }
        val dims = l1.shape().dims4()
        val b = dims.d0
        val t = dims.d1
        val h = dims.d2
        val d = dims.d3
        val elCount = b * t * h * d
        val src = s1.asSlice()
        val cos = s2.asSlice()
        val sin = s3.asSlice()
        val srcOffset = l1.startOffset()
        val cosOffset = l2.startOffset()
        val sinOffset = l3.startOffset()
        val cosDims = l2.dims()
        val isBatchedCos = cosDims.size == 3
        val halfD = d / 2
        val dst = MutableList(elCount) { 0.0 }

        for (bIdx in 0 until b) {
            val cosBOffset = if (isBatchedCos) bIdx * t * halfD else 0
            for (tIdx in 0 until t) {
                val csBase = cosBOffset + tIdx * halfD
                for (hIdx in 0 until h) {
                    val baseIdx = ((bIdx * t + tIdx) * h + hIdx) * d
                    for (i in 0 until halfD) {
                        val x1 = src[srcOffset + baseIdx + i]
                        val x2 = src[srcOffset + baseIdx + halfD + i]
                        val c = cos[cosOffset + csBase + i]
                        val s = sin[sinOffset + csBase + i]
                        dst[baseIdx + i] = x1 * c - x2 * s
                        dst[baseIdx + halfD + i] = x1 * s + x2 * c
                    }
                }
            }
        }
        return Pair(wrapRopeCpuStorage(s1.dtype(), dst), l1.shape())
    }

    override fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        arg3: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Triple<Tensor?, Tensor?, Tensor?> {
        if (arg1.dtype() != res.dtype()) {
            bail("rope-thd bwd dtype mismatch")
        }
        val gradX = ropeThd(gradRes, arg2, arg3.neg())
        return Triple(gradX, null, null)
    }
}

private fun checkRopeArguments(
    b: Int,
    t: Int,
    d: Int,
    cos: Tensor,
    sin: Tensor,
) {
    if (d % 2 != 0) {
        bail("rope requires an even head dimension, got $d")
    }
    if (cos.dims() != sin.dims()) {
        bail("shape mismatch in rope: cos ${cos.shape()} vs sin ${sin.shape()}")
    }
    val validCosShape = cos.dims() == listOf(t, d / 2) || cos.dims() == listOf(b, t, d / 2)
    if (!validCosShape) {
        bail("shape mismatch in rope: x ($b, *, $t, $d) vs cos ${cos.shape()}")
    }
}

/**
 * Contiguous variant of rope embeddings (interleaved pairs).
 */
public fun ropeI(xs: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val dims = xs.dims4()
    checkRopeArguments(dims.d0, dims.d2, dims.d3, cos, sin)
    return xs.applyOp3NoBwd(cos, sin, RopeIOp())
}

public fun ropeISlow(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val dims = x.dims4()
    val bSz = dims.d0
    val nHead = dims.d1
    val seqLen = dims.d2
    val nEmbd = dims.d3
    val cosReshaped = cos
        .reshape(Shape(seqLen, nEmbd / 2, 1))
        .broadcastLeft(Shape(bSz, nHead))
    val sinReshaped = sin
        .reshape(Shape(seqLen, nEmbd / 2, 1))
        .broadcastLeft(Shape(bSz, nHead))
    val xReshaped = x.reshape(Shape(bSz, nHead, seqLen, nEmbd / 2, 2))
    val x0 = xReshaped.narrow(D.Minus1, 0, 1)
    val x1 = xReshaped.narrow(D.Minus1, 1, 1)
    val y0 = (x0.broadcastMul(cosReshaped) - x1.broadcastMul(sinReshaped))
    val y1 = (x0.broadcastMul(sinReshaped) + x1.broadcastMul(cosReshaped))
    return Tensor.cat(listOf(y0, y1), D.Minus1).flattenFrom(D.Minus2)
}

/**
 * Split-half variant of rotary position embeddings for `(B, H, T, D)` tensors.
 */
public fun rope(xs: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val dims = xs.dims4()
    checkRopeArguments(dims.d0, dims.d2, dims.d3, cos, sin)
    return xs.applyOp3NoBwd(cos, sin, RopeOp())
}

public fun ropeSlow(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val nEmbd = x.dims4().d3
    val cosUnsqueezed = cos.unsqueeze(0).unsqueeze(0)
    val sinUnsqueezed = sin.unsqueeze(0).unsqueeze(0)
    val x1 = x.narrow(D.Minus1, 0, nEmbd / 2)
    val x2 = x.narrow(D.Minus1, nEmbd / 2, nEmbd / 2)
    val rotateX = Tensor.cat(listOf(x2.neg(), x1), D.Minus1)
    val repeatCos = Tensor.cat(listOf(cosUnsqueezed, cosUnsqueezed), D.Minus1)
    val repeatSin = Tensor.cat(listOf(sinUnsqueezed, sinUnsqueezed), D.Minus1)
    return x.broadcastMul(repeatCos) + rotateX.broadcastMul(repeatSin)
}

/**
 * Split-half variant of rotary position embeddings for `(B, T, H, D)` tensors.
 */
public fun ropeThd(xs: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val dims = xs.dims4()
    checkRopeArguments(dims.d0, dims.d1, dims.d3, cos, sin)
    return xs.applyOp3NoBwd(cos, sin, RopeThdOp())
}

public fun ropeThdSlow(x: Tensor, cos: Tensor, sin: Tensor): Tensor {
    val nEmbd = x.dims4().d3
    val cosUnsqueezed = cos.unsqueeze(0).unsqueeze(2)
    val sinUnsqueezed = sin.unsqueeze(0).unsqueeze(2)
    val x1 = x.narrow(D.Minus1, 0, nEmbd / 2)
    val x2 = x.narrow(D.Minus1, nEmbd / 2, nEmbd / 2)
    val rotateX = Tensor.cat(listOf(x2.neg(), x1), D.Minus1)
    val repeatCos = Tensor.cat(listOf(cosUnsqueezed, cosUnsqueezed), D.Minus1)
    val repeatSin = Tensor.cat(listOf(sinUnsqueezed, sinUnsqueezed), D.Minus1)
    return x.broadcastMul(repeatCos) + rotateX.broadcastMul(repeatSin)
}
