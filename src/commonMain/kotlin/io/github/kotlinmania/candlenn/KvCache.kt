// port-lint: source kv_cache.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.sliceSet

public class Cache(
    private var allData: Tensor?,
    private val dim: Int,
    private var currentSeqLen: Int,
    private var maxSeqLen: Int,
) {
    public companion object {
        public fun new(dim: Int, maxSeqLen: Int): Cache =
            Cache(
                allData = null,
                dim = dim,
                currentSeqLen = 0,
                maxSeqLen = maxSeqLen,
            )
    }

    public fun dim(): Int = dim

    public fun currentSeqLen(): Int = currentSeqLen

    public fun maxSeqLen(): Int = maxSeqLen

    public fun allData(): Tensor? = allData

    public fun currentData(): Tensor? =
        allData?.narrow(dim, 0, currentSeqLen)

    public fun reset() {
        currentSeqLen = 0
        allData = null
    }

    public fun append(src: Tensor) {
        val seqLen = src.dim(dim)
        // This doesn't seem very idiomatic, but because the creation of the cache is lazy, it's
        // tricky to use `self.allData.getOrInsertWith`.
        if (allData == null) {
            val shapeDims = src.dims().toMutableList()
            shapeDims[dim] = maxSeqLen
            val ad = Tensor.zeros(Shape(shapeDims), src.dtype(), src.device())
            allData = ad
        }
        var ad = allData!!
        if (currentSeqLen + seqLen > maxSeqLen) {
            val nextMaxSeqLen = maxOf(currentSeqLen + seqLen, 2 * maxSeqLen)
            val shapeDims = src.dims().toMutableList()
            shapeDims[dim] = nextMaxSeqLen
            val nextAd = Tensor.zeros(Shape(shapeDims), src.dtype(), src.device())
            nextAd.sliceSet(ad, dim, 0)
            maxSeqLen = nextMaxSeqLen
            allData = nextAd
            ad = nextAd
        }
        ad.sliceSet(src, dim, currentSeqLen)
        currentSeqLen += seqLen
    }
}

public class KvCache(
    private val k: Cache,
    private val v: Cache,
) {
    public companion object {
        public fun new(dim: Int, maxSeqLen: Int): KvCache {
            val k = Cache.new(dim, maxSeqLen)
            val v = Cache.new(dim, maxSeqLen)
            return KvCache(k, v)
        }
    }

    public fun kCache(): Cache = k

    public fun vCache(): Cache = v

    public fun kCacheMut(): Cache = k

    public fun vCacheMut(): Cache = v

    public fun k(): Tensor? = k.currentData()

    public fun v(): Tensor? = v.currentData()

    public fun append(kTensor: Tensor, vTensor: Tensor): Pair<Tensor, Tensor> {
        val kContig = kTensor.contiguous()
        val vContig = vTensor.contiguous()
        k.append(kContig)
        v.append(vContig)
        val outK = k.currentData()!!
        val outV = v.currentData()!!
        return Pair(outK, outV)
    }

    public fun currentSeqLen(): Int = k.currentSeqLen()

    public fun reset() {
        k.reset()
        v.reset()
    }
}

public class RotatingCache(
    private var allData: Tensor?,
    private val dim: Int,
    // `offset` is the current write index in the buffer
    private var offset: Int,
    // The total size of the sequence seen so far.
    private var currentSeqLen: Int,
    // maxSeqLen is the size of the rotating buffer, it is allowed for the full
    // sequence to grow past this limit.
    private val maxSeqLen: Int,
) {
    public companion object {
        public fun new(dim: Int, maxSeqLen: Int): RotatingCache =
            RotatingCache(
                allData = null,
                dim = dim,
                offset = 0,
                currentSeqLen = 0,
                maxSeqLen = maxSeqLen,
            )
    }

    public fun offset(): Int = offset

    public fun dim(): Int = dim

    public fun currentSeqLen(): Int = currentSeqLen

    public fun maxSeqLen(): Int = maxSeqLen

    public fun allData(): Tensor? = allData

    public fun currentData(): Tensor? =
        if (currentSeqLen >= maxSeqLen) {
            allData
        } else {
            allData?.narrow(dim, 0, currentSeqLen)
        }

    public fun reset() {
        offset = 0
        currentSeqLen = 0
        allData = null
    }

    public fun append(src: Tensor): Tensor {
        val seqLen = src.dim(dim)
        // This doesn't seem very idiomatic, but because the creation of the cache is lazy, it's
        // tricky to use `self.allData.getOrInsertWith`.
        if (allData == null) {
            val shapeDims = src.dims().toMutableList()
            shapeDims[dim] = maxSeqLen
            val ad = Tensor.zeros(Shape(shapeDims), src.dtype(), src.device())
            allData = ad
        }
        val ad = allData!!

        currentSeqLen += seqLen
        if (seqLen >= maxSeqLen) {
            val toCopy = src
                .narrow(dim, seqLen - maxSeqLen, maxSeqLen)
                .contiguous()
            ad.sliceSet(toCopy, dim, 0)
            offset = 0
            // Here we return `src` rather than `ad` so that all the past can be used in this
            // step properly.
            return src
        } else {
            val remLen = maxSeqLen - offset
            if (seqLen <= remLen) {
                ad.sliceSet(src.contiguous(), dim, offset)
                offset = (offset + seqLen) % maxSeqLen
            } else {
                // We have to make two copies here as we go over the boundary of the cache.
                if (remLen > 0) {
                    val src1 = src.narrow(dim, 0, remLen).contiguous()
                    ad.sliceSet(src1, dim, offset)
                }
                val src2 = src.narrow(dim, remLen, seqLen - remLen).contiguous()
                ad.sliceSet(src2, dim, 0)
                offset = seqLen - remLen
            }
            return if (currentSeqLen >= maxSeqLen) {
                ad
            } else {
                ad.narrow(dim, 0, currentSeqLen)
            }
        }
    }

    /**
     * Returns the causal mask for a new sequence of length `seqLen` Needs to include the
     * context so the mask has shape `(seqLen, currentSeqLen)` if `currentSeqLen < maxSeqLen`
     * or `(seqLen, maxSeqLen)` otherwise.
     */
    public fun attnMask(seqLen: Int, device: Device): Tensor? {
        if (seqLen == 1) {
            return null
        }
        val mask = if (seqLen < maxSeqLen) {
            val totalSize = if (currentSeqLen + seqLen > maxSeqLen) {
                // In that case, the mask does not have column Length `maxSeqLen`, rather it has
                // two disconnected parts.
                maxSeqLen
            } else {
                currentSeqLen + seqLen
            }
            val maskRows = ArrayList<UByteArray>(seqLen)
            val absOffset = (offset + seqLen) % maxSeqLen
            for (i in 0 until seqLen) {
                val maskRow = UByteArray(totalSize)
                for (j in 0 until totalSize) {
                    // Find what time step column j corresponds to relative to the end of the step.
                    // In the circular buffer, `absOffset` is where the *next* write will happen
                    // after this step completes.
                    val dist = (absOffset + maxSeqLen - 1 - j) % maxSeqLen
                    val isFuture = dist < (seqLen - 1 - i)
                    val isUnwritten = currentSeqLen + seqLen < maxSeqLen && j >= currentSeqLen + seqLen
                    maskRow[j] = if (isFuture || isUnwritten) 1u else 0u
                }
                maskRows.add(maskRow)
            }
            val flat = UByteArray(seqLen * totalSize)
            for (i in 0 until seqLen) {
                maskRows[i].copyInto(flat, i * totalSize)
            }
            Tensor.new(flat, Shape(seqLen, totalSize), device)
        } else {
            // In that case we return `src` directly in `append` so the mask is a standard
            // lower-triangular mask of size `(seqLen, seqLen)`.
            val flat = UByteArray(seqLen * seqLen)
            for (i in 0 until seqLen) {
                for (j in 0 until seqLen) {
                    flat[i * seqLen + j] = if (j > i || i - j >= maxSeqLen) 1u else 0u
                }
            }
            Tensor.new(flat, Shape(seqLen, seqLen), device)
        }
        return mask
    }

    public fun positions(seqLen: Int): List<Int> =
        if (seqLen <= maxSeqLen) {
            val absOffset = (offset + seqLen) % maxSeqLen
            val totalSize = minOf(currentSeqLen + seqLen, maxSeqLen)
            val out = MutableList(totalSize) { 0 }
            for (i in 0 until totalSize) {
                val idx = (absOffset + maxSeqLen - 1 - i) % maxSeqLen
                out[idx] = currentSeqLen + seqLen - 1 - i
            }
            out
        } else {
            (currentSeqLen until (currentSeqLen + seqLen)).toList()
        }
}

public class RotatingKvCache(
    private val k: RotatingCache,
    private val v: RotatingCache,
) {
    public companion object {
        public fun new(dim: Int, maxSeqLen: Int): RotatingKvCache {
            val k = RotatingCache.new(dim, maxSeqLen)
            val v = RotatingCache.new(dim, maxSeqLen)
            return RotatingKvCache(k, v)
        }
    }

    public fun kCache(): RotatingCache = k

    public fun vCache(): RotatingCache = v

    public fun kCacheMut(): RotatingCache = k

    public fun vCacheMut(): RotatingCache = v

    public fun k(): Tensor? = k.currentData()

    public fun v(): Tensor? = v.currentData()

    public fun append(kTensor: Tensor, vTensor: Tensor): Pair<Tensor, Tensor> {
        val outK = k.append(kTensor)
        val outV = v.append(vTensor)
        return Pair(outK, outV)
    }

    public fun offset(): Int = k.offset()

    public fun currentSeqLen(): Int = k.currentSeqLen()

    public fun maxSeqLen(): Int = k.maxSeqLen()

    public fun attnMask(seqLen: Int, device: Device): Tensor? =
        k.attnMask(seqLen, device)

    public fun positions(seqLen: Int): List<Int> =
        k.positions(seqLen)

    public fun reset() {
        k.reset()
        v.reset()
    }
}

public class IndicesAndMask(
    internal val indices: Tensor,
    internal val mask: Tensor,
) {
    public fun mask(): Tensor = mask
}

public class ScatteredKvCache(
    private val k: Tensor,
    private val v: Tensor,
    private val context: Int,
) {
    public fun append(
        kSrc: Tensor,
        vSrc: Tensor,
        iam: IndicesAndMask,
    ): Pair<Tensor, Tensor> {
        if (context <= kSrc.dim(2)) {
            return Pair(kSrc, vSrc)
        }
        val indices = iam.indices.unsqueeze(2).unsqueeze(1)
        val indicesBroadcast = indices.broadcastAs(kSrc.shape()).contiguous()
        k.scatterSet(indicesBroadcast, kSrc.contiguous(), 2)
        v.scatterSet(indicesBroadcast, vSrc.contiguous(), 2)
        return Pair(k, v)
    }

    public fun k(): Tensor = k

    public fun v(): Tensor = v
}

public class ScatteredCacheBuilder internal constructor(
    private val context: Int,
    // The current position in the stream, modularized by context
    private val positions: MutableList<Int>,
    // The index where the next element will be written
    private val indices: MutableList<Int>,
    private val dtype: DType,
    private val device: Device,
) {
    public companion object {
        public fun new(batchSize: Int, context: Int, dtype: DType, device: Device): ScatteredCacheBuilder =
            ScatteredCacheBuilder(
                context = context,
                positions = MutableList(batchSize) { 0 },
                indices = MutableList(batchSize) { 0 },
                dtype = dtype,
                device = device,
            )
    }

    public fun makeCache(numHeads: Int, headDim: Int): ScatteredKvCache {
        val batchSize = batchSize()
        val shape = Shape(batchSize, numHeads, context, headDim)
        val k = Tensor.zeros(shape, dtype, device)
        val v = Tensor.zeros(shape, dtype, device)
        return ScatteredKvCache(k, v, context)
    }

    public fun positions(): List<Int> = positions

    public fun reset() {
        positions.fill(0)
        indices.fill(0)
    }

    public fun batchSize(): Int = positions.size

    public fun resetBatchIndex(batchIndex: Int) {
        positions[batchIndex] = 0
        indices[batchIndex] = 0
    }

    public fun indicesAndMask(
        seqLen: Int,
        batchMask: List<Boolean>,
    ): IndicesAndMask {
        // mask shape is (b, h, t, k)
        val context = this.context
        if (this.context <= seqLen) {
            return indicesAndMaskAbs(seqLen, batchMask)
        }
        val attentionMasks = ArrayList<List<Float>>(batchSize())
        val cacheIndices = ArrayList<List<UInt>>(batchSize())
        for ((batchI, isMasked) in batchMask.withIndex()) {
            if (!isMasked) {
                val masks = List(seqLen) { List(context) { 0.0f } }
                val outIndices = List(seqLen) { indices[batchI].toUInt() }
                attentionMasks.addAll(masks)
                cacheIndices.add(outIndices)
                continue
            }
            val startIndex = indices[batchI]
            val startPos = positions[batchI]
            val masks = ArrayList<List<Float>>(seqLen)
            val outIndices = ArrayList<UInt>(seqLen)
            val allPos = MutableList(context) { Int.MAX_VALUE }
            if (startPos < context) {
                for (i in 0 until startPos) {
                    allPos[i] = i
                }
            } else {
                val offset = startPos - startIndex
                for (i in 0 until context) {
                    allPos[i] = if (i < startIndex) {
                        i + offset
                    } else {
                        i + offset - context
                    }
                }
            }
            for (seqI in 0 until seqLen) {
                val index = indices[batchI]
                allPos[index] = seqI + startPos
                outIndices.add(index.toUInt())
                positions[batchI] += 1
                indices[batchI] += 1
                if (indices[batchI] >= this.context) {
                    indices[batchI] = 0
                }
            }
            for (seqI in 0 until seqLen) {
                val myPos = seqI + startPos
                val myMask = allPos.map { pos ->
                    if (pos <= myPos) 0.0f else Float.NEGATIVE_INFINITY
                }
                masks.add(myMask)
            }

            attentionMasks.addAll(masks)
            cacheIndices.add(outIndices)
        }
        // Flattening the attention mask.
        val flatMask = FloatArray(batchSize() * seqLen * context)
        var offset = 0
        for (row in attentionMasks) {
            for (v in row) {
                flatMask[offset++] = v
            }
        }
        val mask = Tensor.new(flatMask, Shape(batchSize(), 1, seqLen, context), device)
            .toDtype(dtype)
        val flatIndices = UIntArray(batchSize() * seqLen)
        var idxOffset = 0
        for (row in cacheIndices) {
            for (v in row) {
                flatIndices[idxOffset++] = v
            }
        }
        val indicesTensor = Tensor.new(flatIndices, Shape(batchSize(), seqLen), device)
        return IndicesAndMask(indicesTensor, mask)
    }

    public fun device(): Device = device

    private fun indicesAndMaskAbs(
        seqLen: Int,
        batchMask: List<Boolean>,
    ): IndicesAndMask {
        val maskTensor = getMaskAbs(seqLen, seqLen)
        val cacheIndices = ArrayList<List<UInt>>(batchSize())
        for ((batchI, isMasked) in batchMask.withIndex()) {
            if (!isMasked) {
                val outIndices = List(seqLen) { indices[batchI].toUInt() }
                cacheIndices.add(outIndices)
                continue
            }
            val outIndices = ArrayList<UInt>(seqLen)
            for (seqI in 0 until seqLen) {
                val index = indices[batchI] + (seqI * 0)
                outIndices.add(index.toUInt())
                positions[batchI] += 1
                indices[batchI] += 1
                if (indices[batchI] >= this.context) {
                    indices[batchI] = 0
                }
            }
            cacheIndices.add(outIndices)
        }
        val flatIndices = UIntArray(batchSize() * seqLen)
        var idxOffset = 0
        for (row in cacheIndices) {
            for (v in row) {
                flatIndices[idxOffset++] = v
            }
        }
        val indicesTensor = Tensor.new(flatIndices, Shape(batchSize(), seqLen), device)
        return IndicesAndMask(indicesTensor, maskTensor)
    }

    private fun getMaskAbs(size1: Int, size2: Int): Tensor {
        val context = this.context
        val mask = FloatArray(size1 * size2)
        for (i in 0 until size1) {
            for (j in 0 until size2) {
                mask[i * size2 + j] = if (size1 + j > size2 + i || size1 + j + context < size2 + i) {
                    Float.NEGATIVE_INFINITY
                } else {
                    0.0f
                }
            }
        }
        return Tensor.fromSlice(mask, Shape(size1, size2), device).toDtype(dtype)
    }
}

public class ConcatKvCache(
    private val dim: Int,
    private var k: Tensor?,
    private var v: Tensor?,
) {
    public companion object {
        /**
         * Create a new `ConcatKvCache` that will concatenate along the given `dim`.
         */
        public fun new(dim: Int): ConcatKvCache =
            ConcatKvCache(dim, null, null)
    }

    public fun dim(): Int = dim

    public fun isEmpty(): Boolean = k == null

    public fun k(): Tensor? = k

    public fun v(): Tensor? = v

    public fun kMut(): Tensor? = k

    public fun vMut(): Tensor? = v

    public fun intoInner(): Pair<Tensor, Tensor>? {
        val curK = k
        val curV = v
        return if (curK != null && curV != null) Pair(curK, curV) else null
    }

    public fun currentSeqLen(): Int =
        k?.dims()?.getOrNull(dim) ?: 0

    /**
     * Append a key and value tensor to the cache, returning the updated key and value tensors.
     */
    public fun append(kTensor: Tensor, vTensor: Tensor): Pair<Tensor, Tensor> {
        val kDetached = kTensor.contiguous().detach()
        val vDetached = vTensor.contiguous().detach()
        val nextK = when (val prevK = k) {
            null -> kDetached
            else -> Tensor.cat(listOf(prevK, kDetached), dim).detach()
        }
        val nextV = when (val prevV = v) {
            null -> vDetached
            else -> Tensor.cat(listOf(prevV, vDetached), dim).detach()
        }
        k = nextK
        v = nextV
        return Pair(nextK, nextV)
    }

    public fun reset() {
        k = null
        v = null
    }
}

public class InterleavedKvCache(
    private var kv: Tensor?,
    private val headDim: Int,
) {
    public companion object {
        public fun new(headDim: Int): InterleavedKvCache =
            InterleavedKvCache(null, headDim)
    }

    public fun currentSeqLen(): Int =
        kv?.dims()?.firstOrNull() ?: 0

    public fun isEmpty(): Boolean = kv == null

    public fun append(k: Tensor, v: Tensor): Tensor {
        val kSeq = k.squeeze(0).transpose(0, 1).contiguous().detach()
        val vSeq = v.squeeze(0).transpose(0, 1).contiguous().detach()
        val kvNew = Tensor.cat(listOf(kSeq, vSeq), 2).detach()
        val nextKv = when (val prevKv = kv) {
            null -> kvNew
            else -> Tensor.cat(listOf(prevKv, kvNew), 0).detach()
        }
        kv = nextKv
        return nextKv
    }

    public fun kv(): Tensor? = kv

    public fun headDim(): Int = headDim

    public fun reset() {
        kv = null
    }
}

public class RawInterleavedKvCache(
    private var buf: FloatArray,
    private val hKv: Int,
    private val d: Int,
    private val posStride: Int,
    private var len: Int,
) {
    public companion object {
        public fun new(hKv: Int, d: Int, maxSeq: Int): RawInterleavedKvCache {
            val posStride = hKv * 2 * d
            return RawInterleavedKvCache(
                buf = FloatArray(maxSeq * posStride),
                hKv = hKv,
                d = d,
                posStride = posStride,
                len = 0,
            )
        }
    }

    public fun len(): Int = len

    public fun isEmpty(): Boolean = len == 0

    public fun writeKv(kFlat: FloatArray, vFlat: FloatArray) {
        val pos = len
        val base = pos * posStride
        if (base + posStride > buf.size) {
            buf = buf.copyOf(buf.size * 2)
        }
        for (h in 0 until hKv) {
            val kSrc = h * d
            val vSrc = h * d
            val dst = base + h * 2 * d
            kFlat.copyInto(buf, dst, kSrc, kSrc + d)
            vFlat.copyInto(buf, dst + d, vSrc, vSrc + d)
        }
        len += 1
    }

    public fun writeKvBatch(kFlat: FloatArray, vFlat: FloatArray, seqLen: Int) {
        val hd = hKv * d
        for (s in 0 until seqLen) {
            val kRow = kFlat.copyOfRange(s * hd, (s + 1) * hd)
            val vRow = vFlat.copyOfRange(s * hd, (s + 1) * hd)
            writeKv(kRow, vRow)
        }
    }

    public fun data(): FloatArray =
        buf.copyOfRange(0, len * posStride)

    public fun reset() {
        len = 0
    }

    public fun hKv(): Int = hKv

    public fun d(): Int = d
}

public fun ropeIInplace(
    data: FloatArray,
    cos: FloatArray,
    sin: FloatArray,
    numHeads: Int,
    d: Int,
) {
    val halfD = d / 2
    for (h in 0 until numHeads) {
        val base = h * d
        for (i in 0 until halfD) {
            val a = data[base + 2 * i]
            val b = data[base + 2 * i + 1]
            data[base + 2 * i] = a * cos[i] - b * sin[i]
            data[base + 2 * i + 1] = b * cos[i] + a * sin[i]
        }
    }
}
