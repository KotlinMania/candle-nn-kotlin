// port-lint: source var_builder.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.BufferedSafetensors
import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.MmapedSafetensors
import io.github.kotlinmania.candle.NpzTensors
import io.github.kotlinmania.candle.PthTensors
import io.github.kotlinmania.candle.SafeTensors
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.SliceSafetensors
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.tryFrom

/**
 * A trait that defines how tensor data is retrieved.
 */
public interface Backend<H> {
    public fun defaultHints(): H

    public fun get(
        s: Shape,
        name: String,
        h: H,
        dtype: DType,
        dev: Device,
    ): Tensor

    public fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor

    public fun containsTensor(name: String): Boolean
}

public interface SimpleBackend : Backend<Init> {
    override fun defaultHints(): Init = Init.default()
}

internal class TensorData<B : Backend<H>, H>(
    val backend: B,
    val device: Device,
    val dtype: DType,
)

/**
 * A structure used to retrieve variables, these variables can either come from storage or be
 * generated via some form of initialization.
 */
internal open class VarBuilderArgs<B : Backend<H>, H> internal constructor(
    internal val data: TensorData<B, H>,
    internal val pathList: List<String>,
    public val dtype: DType,
) {
    public open fun clone(): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = data,
            pathList = pathList.toList(),
            dtype = dtype,
        )

    /**
     * Returns the prefix of the [VarBuilder].
     */
    public fun prefix(): String = pathList.joinToString(".")

    /**
     * Returns a new [VarBuilderArgs] using the root path.
     */
    public open fun root(): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = data,
            pathList = emptyList(),
            dtype = dtype,
        )

    /**
     * Returns a new [VarBuilderArgs] with the prefix set to [prefix].
     */
    public open fun setPrefix(prefix: String): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = data,
            pathList = listOf(prefix),
            dtype = dtype,
        )

    /**
     * Return a new [VarBuilderArgs] adding [s] to the current prefix.
     */
    public open fun pushPrefix(s: String): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = data,
            pathList = pathList + s,
            dtype = dtype,
        )

    /**
     * Short alias for [pushPrefix].
     */
    public open fun pp(s: String): VarBuilderArgs<B, H> = pushPrefix(s)

    public open fun pp(s: Int): VarBuilderArgs<B, H> = pushPrefix(s.toString())

    /**
     * The device used by default.
     */
    public fun device(): Device = data.device

    /**
     * The dtype used by default.
     */
    public fun dtype(): DType = dtype

    /**
     * Clone the [VarBuilderArgs] tweaking its dtype.
     */
    public open fun toDtype(dtype: DType): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = data,
            pathList = pathList.toList(),
            dtype = dtype,
        )

    internal fun path(tensorName: String): String =
        if (pathList.isEmpty()) {
            tensorName
        } else {
            "${pathList.joinToString(".")}.$tensorName"
        }

    /**
     * Returns true only if a tensor with the passed in name is available.
     */
    public fun containsTensor(tensorName: String): Boolean {
        val fullPath = path(tensorName)
        return data.backend.containsTensor(fullPath)
    }

    /**
     * Retrieve the tensor associated with the given name at the current path.
     */
    public fun getWithHints(
        s: Shape,
        name: String,
        hints: H,
    ): Tensor = getWithHintsDtype(s, name, hints, dtype)

    public fun getWithHints(
        s: Int,
        name: String,
        hints: H,
    ): Tensor = getWithHints(Shape(s), name, hints)

    public fun getWithHints(
        s: Pair<Int, Int>,
        name: String,
        hints: H,
    ): Tensor = getWithHints(Shape(s.first, s.second), name, hints)

    public fun getWithHints(
        s: Triple<Int, Int, Int>,
        name: String,
        hints: H,
    ): Tensor = getWithHints(Shape(s.first, s.second, s.third), name, hints)

    public fun getWithHints(
        s: List<Int>,
        name: String,
        hints: H,
    ): Tensor = getWithHints(Shape(s), name, hints)

    /**
     * Retrieve the tensor associated with the given name at the current path.
     */
    public fun get(
        s: Shape,
        name: String,
    ): Tensor = getWithHints(s, name, data.backend.defaultHints())

    public fun get(
        s: Int,
        name: String,
    ): Tensor = get(Shape(s), name)

    public fun get(
        s: Pair<Int, Int>,
        name: String,
    ): Tensor = get(Shape(s.first, s.second), name)

    public fun get(
        s: Triple<Int, Int, Int>,
        name: String,
    ): Tensor = get(Shape(s.first, s.second, s.third), name)

    public fun get(
        s: List<Int>,
        name: String,
    ): Tensor = get(Shape(s), name)

    /**
     * Retrieve the tensor associated with the given name at the current path.
     */
    public fun getUnchecked(name: String): Tensor = getUncheckedDtype(name, data.dtype)

    /**
     * Retrieve the tensor associated with the given name and dtype at the current path.
     */
    public fun getUncheckedDtype(
        name: String,
        dtype: DType,
    ): Tensor {
        val fullPath = path(name)
        return data.backend.getUnchecked(fullPath, dtype, data.device)
    }

    /**
     * Retrieve the tensor associated with the given name and dtype at the current path.
     */
    public fun getWithHintsDtype(
        s: Shape,
        name: String,
        hints: H,
        dtype: DType,
    ): Tensor {
        val fullPath = path(name)
        return data.backend.get(s, fullPath, hints, dtype, data.device)
    }

    /**
     * Set the device of the [VarBuilderArgs].
     */
    public open fun setDevice(device: Device): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = TensorData(data.backend, device, data.dtype),
            pathList = pathList.toList(),
            dtype = dtype,
        )

    /**
     * Set the dtype of the [VarBuilderArgs].
     */
    public open fun setDtype(dtype: DType): VarBuilderArgs<B, H> =
        VarBuilderArgs(
            data = TensorData(data.backend, data.device, dtype),
            pathList = pathList.toList(),
            dtype = dtype,
        )

    public companion object {
        public fun <B : Backend<H>, H> newWithArgs(
            backend: B,
            dtype: DType,
            dev: Device,
        ): VarBuilderArgs<B, H> =
            VarBuilderArgs(
                data = TensorData(backend, dev, dtype),
                pathList = emptyList(),
                dtype = dtype,
            )
    }
}

internal object Zeros : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor = Tensor.zeros(s, dtype, dev)

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor =
        throw CandleException.Msg(
            "`Zeros` requires a shape for tensor retrieval, use `get` instead of `getUnchecked`",
        )

    override fun containsTensor(name: String): Boolean = true
}

internal class MapBackend(
    private val tensors: Map<String, Tensor>,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val tensor =
            tensors[name]
                ?: throw CandleException.CannotFindTensor(path = name)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor.toDevice(dev).toDtype(dtype)
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val tensor =
            tensors[name]
                ?: throw CandleException.CannotFindTensor(path = name)
        return tensor.toDevice(dev).toDtype(dtype)
    }

    override fun containsTensor(name: String): Boolean = tensors.containsKey(name)
}

internal class VarMapBackend(
    private val varMap: VarMap,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor = varMap.get(s, name, h, dtype, dev)

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor =
        throw CandleException.Msg("`getUnchecked` does not make sense for `VarMap`, use `get`.")

    override fun containsTensor(name: String): Boolean = varMap.data().containsKey(name)
}

public class SafeTensorWithRouting(
    private val routing: Map<String, Int>,
    private val safetensors: List<SafeTensors>,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val index =
            routing[name]
                ?: throw CandleException.CannotFindTensor(path = name)
        val tensor =
            safetensors[index]
                .tensor(name)
                .getOrThrow()
                .load(dev)
                .getOrThrow()
                .toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val index =
            routing[name]
                ?: throw CandleException.CannotFindTensor(path = name)
        return safetensors[index]
            .tensor(name)
            .getOrThrow()
            .load(dev)
            .getOrThrow()
            .toDtype(dtype)
    }

    override fun containsTensor(name: String): Boolean = routing.containsKey(name)
}

internal class NpzBackend(
    private val npz: NpzTensors,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val raw =
            npz.get(name)
                ?: throw CandleException.CannotFindTensor(path = name)
        val tensor = raw.toDevice(dev).toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val raw =
            npz.get(name)
                ?: throw CandleException.CannotFindTensor(path = name)
        return raw.toDevice(dev).toDtype(dtype)
    }

    override fun containsTensor(name: String): Boolean =
        try {
            npz.get(name) != null
        } catch (_: Throwable) {
            false
        }
}

internal class PthBackend(
    private val pth: PthTensors,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val raw =
            pth.get(name).getOrThrow()
                ?: throw CandleException.CannotFindTensor(path = name)
        val tensor = raw.toDevice(dev).toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val raw =
            pth.get(name).getOrThrow()
                ?: throw CandleException.CannotFindTensor(path = name)
        return raw.toDevice(dev).toDtype(dtype)
    }

    override fun containsTensor(name: String): Boolean =
        pth.get(name).unwrapOrElse { null } != null
}

internal class MmapedSafetensorsBackend(
    val inner: MmapedSafetensors,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)

    override fun containsTensor(name: String): Boolean = inner.get(name).isOk
}

internal class BufferedSafetensorsBackend(
    private val inner: BufferedSafetensors,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)

    override fun containsTensor(name: String): Boolean = inner.get(name).isOk
}

internal class SliceSafetensorsBackend(
    private val inner: SliceSafetensors,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (s.elemCount() < 0 && h == Init.Const(0.0)) {
            bail("invalid shape $s")
        }
        val tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)
        if (tensor.shape() != s) {
            throw CandleException.UnexpectedShape(
                msg = "shape mismatch for $name",
                expected = s,
                got = tensor.shape(),
            )
        }
        return tensor
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor = inner.load(name, dev).getOrThrow().toDtype(dtype)

    override fun containsTensor(name: String): Boolean = inner.get(name).isOk
}

/**
 * This interface specifies a way to rename the queried names into names that are stored in an inner
 * [VarBuilder].
 */
public fun interface Renamer {
    public fun rename(v: String): String
}

public class Rename(
    private val inner: VarBuilder,
    private val renamer: Renamer,
) : SimpleBackend {
    override fun get(
        s: Shape,
        name: String,
        h: Init,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val renamed = renamer.rename(name)
        return inner.getWithHintsDtype(s, renamed, h, dtype).toDevice(dev)
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor {
        val renamed = renamer.rename(name)
        return inner.getUncheckedDtype(renamed, dtype).toDevice(dev)
    }

    override fun containsTensor(name: String): Boolean {
        val renamed = renamer.rename(name)
        return inner.containsTensor(renamed)
    }

    public companion object {
        public fun new(
            inner: VarBuilder,
            renamer: Renamer,
        ): Rename = Rename(inner, renamer)
    }
}

/**
 * A simple [VarBuilder] used to retrieve variables from a model checkpoint or [VarMap].
 */
public class VarBuilder internal constructor(
    private val inner: VarBuilderArgs<SimpleBackend, Init>,
) {
    internal val dtype: DType get() = inner.dtype

    public fun clone(): VarBuilder = VarBuilder(inner.clone())

    public fun prefix(): String = inner.prefix()

    public fun root(): VarBuilder = VarBuilder(inner.root())

    public fun setPrefix(prefix: String): VarBuilder = VarBuilder(inner.setPrefix(prefix))

    public fun pushPrefix(s: String): VarBuilder = VarBuilder(inner.pushPrefix(s))

    public fun pp(s: String): VarBuilder = VarBuilder(inner.pp(s))

    public fun pp(s: Int): VarBuilder = VarBuilder(inner.pp(s))

    public fun device(): Device = inner.device()

    public fun dtype(): DType = inner.dtype()

    public fun toDtype(dtype: DType): VarBuilder = VarBuilder(inner.toDtype(dtype))

    public fun containsTensor(tensorName: String): Boolean = inner.containsTensor(tensorName)

    public fun getWithHints(
        s: Shape,
        name: String,
        hints: Init,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun getWithHints(
        s: Int,
        name: String,
        hints: Init,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun getWithHints(
        s: Pair<Int, Int>,
        name: String,
        hints: Init,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun getWithHints(
        s: Triple<Int, Int, Int>,
        name: String,
        hints: Init,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun getWithHints(
        s: List<Int>,
        name: String,
        hints: Init,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun get(
        s: Shape,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun get(
        s: Int,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun get(
        s: Pair<Int, Int>,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun get(
        s: Triple<Int, Int, Int>,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun get(
        s: List<Int>,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun getUnchecked(name: String): Tensor = inner.getUnchecked(name)

    public fun getUncheckedDtype(
        name: String,
        dtype: DType,
    ): Tensor = inner.getUncheckedDtype(name, dtype)

    public fun getWithHintsDtype(
        s: Shape,
        name: String,
        hints: Init,
        dtype: DType,
    ): Tensor = inner.getWithHintsDtype(s, name, hints, dtype)

    public fun setDevice(device: Device): VarBuilder = VarBuilder(inner.setDevice(device))

    public fun setDtype(dtype: DType): VarBuilder = VarBuilder(inner.setDtype(dtype))

    public fun renameF(f: Renamer): VarBuilder = rename(f)

    public fun rename(renamer: Renamer): VarBuilder {
        val currentDtype = dtype()
        val currentDevice = device()
        val currentPath = inner.pathList.toList()
        val backend = Rename.new(this, renamer)
        return VarBuilder(
            VarBuilderArgs(
                data = TensorData(backend, currentDevice, currentDtype),
                pathList = currentPath,
                dtype = currentDtype,
            ),
        )
    }

    public companion object {
        /**
         * Initializes a [VarBuilder] using a custom backend.
         */
        public fun fromBackend(
            backend: SimpleBackend,
            dtype: DType,
            device: Device,
        ): VarBuilder =
            VarBuilder(
                VarBuilderArgs.newWithArgs(backend, dtype, device),
            )

        /**
         * Initializes a [VarBuilder] that uses zeros for any tensor.
         */
        public fun zeros(
            dtype: DType,
            dev: Device,
        ): VarBuilder = fromBackend(Zeros, dtype, dev)

        /**
         * Initializes a [VarBuilder] that retrieves tensors stored in a map.
         */
        public fun fromTensors(
            ts: Map<String, Tensor>,
            dtype: DType,
            dev: Device,
        ): VarBuilder = fromBackend(MapBackend(ts), dtype, dev)

        /**
         * Initializes a [VarBuilder] using a [VarMap].
         */
        public fun fromVarmap(
            varmap: VarMap,
            dtype: DType,
            dev: Device,
        ): VarBuilder = fromBackend(VarMapBackend(varmap.clone()), dtype, dev)

        /**
         * Initializes a [VarBuilder] that retrieves tensors stored in a collection of safetensors
         * files.
         */
        public fun fromMmapedSafetensors(
            paths: List<String>,
            dtype: DType,
            dev: Device,
        ): VarBuilder {
            val tensors = MmapedSafetensors.multi(paths).getOrThrow()
            return fromBackend(MmapedSafetensorsBackend(tensors), dtype, dev)
        }

        /**
         * Initializes a [VarBuilder] from a binary buffer in the safetensor format.
         */
        public fun fromBufferedSafetensors(
            data: ByteArray,
            dtype: DType,
            dev: Device,
        ): VarBuilder {
            val tensors = BufferedSafetensors.new(data).getOrThrow()
            return fromBackend(BufferedSafetensorsBackend(tensors), dtype, dev)
        }

        /**
         * Initializes a [VarBuilder] from a binary slice in the safetensor format.
         */
        public fun fromSliceSafetensors(
            data: ByteArray,
            dtype: DType,
            dev: Device,
        ): VarBuilder {
            val tensors = SliceSafetensors.new(data).getOrThrow()
            return fromBackend(SliceSafetensorsBackend(tensors), dtype, dev)
        }

        /**
         * Initializes a [VarBuilder] that retrieves tensors stored in a numpy npz file.
         */
        public fun fromNpz(
            p: String,
            dtype: DType,
            dev: Device,
        ): VarBuilder {
            val npz = NpzTensors.new(p)
            return fromBackend(NpzBackend(npz), dtype, dev)
        }

        /**
         * Initializes a [VarBuilder] that retrieves tensors stored in a pytorch pth file.
         */
        public fun fromPth(
            p: String,
            dtype: DType,
            dev: Device,
        ): VarBuilder {
            val pth = PthTensors.new(p, null).getOrThrow()
            return fromBackend(PthBackend(pth), dtype, dev)
        }

        /**
         * Initializes a [VarBuilder] that retrieves tensors stored in a pytorch pth file with a state key.
         */
        public fun fromPthWithState(
            p: String,
            dtype: DType,
            stateKey: String,
            dev: Device,
        ): VarBuilder {
            val pth = PthTensors.new(p, stateKey).getOrThrow()
            return fromBackend(PthBackend(pth), dtype, dev)
        }
    }
}

public data class Shard(
    public val dim: Int = 0,
    public val rank: Int = 0,
    public val worldSize: Int = 1,
) {
    public companion object {
        public fun default(): Shard = Shard(0, 0, 1)
    }
}

public class ShardedSafeTensors(
    internal val inner: MmapedSafetensors,
) : Backend<Shard> {
    override fun defaultHints(): Shard = Shard.default()

    override fun get(
        s: Shape,
        name: String,
        h: Shard,
        dtype: DType,
        dev: Device,
    ): Tensor {
        if (h.worldSize == 1) {
            return MmapedSafetensorsBackend(inner).get(s, name, Init.default(), dtype, dev)
        }
        val dim = h.dim
        val rank = h.rank
        val worldSize = h.worldSize
        val view = inner.get(name).getOrThrow()
        val viewDtype = tryFrom(view.wireDtype).getOrThrow()
        val shape = view.tensorShape.toMutableList()
        val size = shape[dim]
        if (size % worldSize != 0) {
            throw CandleException.ShapeMismatchSplit(
                shape = Shape(shape),
                dim = dim,
                nParts = worldSize,
            )
        }
        val blockSize = size / worldSize
        val start = rank * blockSize
        val fullTensor = view.load(dev).getOrThrow()
        return fullTensor.narrow(dim, start, blockSize).toDtype(dtype)
    }

    override fun getUnchecked(
        name: String,
        dtype: DType,
        dev: Device,
    ): Tensor =
        throw CandleException.Msg(
            "`getUnchecked` does not make sense for `ShardedSafeTensors`, use `get`.",
        )

    override fun containsTensor(name: String): Boolean = inner.get(name).isOk

    public companion object {
        public fun varBuilder(
            paths: List<String>,
            dtype: DType,
            dev: Device,
        ): ShardedVarBuilder {
            val tensors = MmapedSafetensors.multi(paths).getOrThrow()
            val backend = ShardedSafeTensors(tensors)
            return ShardedVarBuilder(VarBuilderArgs.newWithArgs(backend, dtype, dev))
        }
    }
}

public class ShardedVarBuilder internal constructor(
    private val inner: VarBuilderArgs<ShardedSafeTensors, Shard>,
) {
    internal val dtype: DType get() = inner.dtype

    public fun clone(): ShardedVarBuilder = ShardedVarBuilder(inner.clone())

    public fun prefix(): String = inner.prefix()

    public fun root(): ShardedVarBuilder = ShardedVarBuilder(inner.root())

    public fun setPrefix(prefix: String): ShardedVarBuilder = ShardedVarBuilder(inner.setPrefix(prefix))

    public fun pushPrefix(s: String): ShardedVarBuilder = ShardedVarBuilder(inner.pushPrefix(s))

    public fun pp(s: String): ShardedVarBuilder = ShardedVarBuilder(inner.pp(s))

    public fun pp(s: Int): ShardedVarBuilder = ShardedVarBuilder(inner.pp(s))

    public fun device(): Device = inner.device()

    public fun dtype(): DType = inner.dtype()

    public fun toDtype(dtype: DType): ShardedVarBuilder = ShardedVarBuilder(inner.toDtype(dtype))

    public fun containsTensor(tensorName: String): Boolean = inner.containsTensor(tensorName)

    public fun getWithHints(
        s: Shape,
        name: String,
        hints: Shard,
    ): Tensor = inner.getWithHints(s, name, hints)

    public fun get(
        s: Shape,
        name: String,
    ): Tensor = inner.get(s, name)

    public fun getUnchecked(name: String): Tensor = inner.getUnchecked(name)

    public fun getUncheckedDtype(
        name: String,
        dtype: DType,
    ): Tensor = inner.getUncheckedDtype(name, dtype)

    public fun getWithHintsDtype(
        s: Shape,
        name: String,
        hints: Shard,
        dtype: DType,
    ): Tensor = inner.getWithHintsDtype(s, name, hints, dtype)

    public fun setDevice(device: Device): ShardedVarBuilder = ShardedVarBuilder(inner.setDevice(device))

    public fun setDtype(dtype: DType): ShardedVarBuilder = ShardedVarBuilder(inner.setDtype(dtype))
}
