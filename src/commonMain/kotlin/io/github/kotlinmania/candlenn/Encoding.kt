// port-lint: source encoding.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

private fun extractLongIndices(indices: Tensor): LongArray {
    val flat = indices.flattenAll()
    return when (val dtype = flat.dtype()) {
        DType.U8 -> {
            val vals = flat.toVec1<Double>()
            LongArray(vals.size) { vals[it].toLong() and 0xFFL }
        }
        DType.U32 -> {
            val vals = flat.toVec1<Double>()
            LongArray(vals.size) { vals[it].toLong() and 0xFFFFFFFFL }
        }
        DType.I64 -> flat.toVec1I64()
        else -> bail("one-hot: unsupported data type $dtype, expected U8, U32, or I64")
    }
}

private fun <T> setAtIndex(
    value: Long,
    offset: Int,
    depth: Int,
    v: MutableList<T>,
    onValue: T,
) {
    // Skip for an entire row of offValues
    if (value == -1L) {
        return
    }
    if (value < -1L) {
        bail("one-hot: invalid negative index value $value, expected a positive index value or -1")
    }
    val idxVal = value.toInt()
    if (idxVal >= depth) {
        bail("one-hot: index value $idxVal exceeds depth $depth")
    }
    val idx = offset + idxVal
    if (idx >= v.size) {
        bail("one-hot: index out of bounds $idx, len ${v.size}")
    }
    v[idx] = onValue
}

/**
 * One-hot/cold encoding for 32-bit float values.
 */
public fun oneHot(
    indices: Tensor,
    depth: Int,
    onValue: Float,
    offValue: Float,
): Tensor {
    val targetShapeDims = indices.dims().toMutableList()
    targetShapeDims.add(depth)
    val flatIndices = extractLongIndices(indices)
    val out = MutableList(depth * flatIndices.size) { offValue }
    for (i in flatIndices.indices) {
        setAtIndex(flatIndices[i], i * depth, depth, out, onValue)
    }
    return Tensor.new(out.toFloatArray(), Shape(targetShapeDims), indices.device())
}

/**
 * One-hot/cold encoding for 64-bit float values.
 */
public fun oneHot(
    indices: Tensor,
    depth: Int,
    onValue: Double,
    offValue: Double,
): Tensor {
    val targetShapeDims = indices.dims().toMutableList()
    targetShapeDims.add(depth)
    val flatIndices = extractLongIndices(indices)
    val out = MutableList(depth * flatIndices.size) { offValue }
    for (i in flatIndices.indices) {
        setAtIndex(flatIndices[i], i * depth, depth, out, onValue)
    }
    return Tensor.new(out.toDoubleArray(), Shape(targetShapeDims), indices.device())
}

/**
 * One-hot/cold encoding for unsigned 8-bit integer values.
 */
public fun oneHot(
    indices: Tensor,
    depth: Int,
    onValue: UByte,
    offValue: UByte,
): Tensor {
    val targetShapeDims = indices.dims().toMutableList()
    targetShapeDims.add(depth)
    val flatIndices = extractLongIndices(indices)
    val out = MutableList(depth * flatIndices.size) { offValue }
    for (i in flatIndices.indices) {
        setAtIndex(flatIndices[i], i * depth, depth, out, onValue)
    }
    return Tensor.new(out.toUByteArray(), Shape(targetShapeDims), indices.device())
}

/**
 * One-hot/cold encoding for unsigned 32-bit integer values.
 */
public fun oneHot(
    indices: Tensor,
    depth: Int,
    onValue: UInt,
    offValue: UInt,
): Tensor {
    val targetShapeDims = indices.dims().toMutableList()
    targetShapeDims.add(depth)
    val flatIndices = extractLongIndices(indices)
    val out = MutableList(depth * flatIndices.size) { offValue }
    for (i in flatIndices.indices) {
        setAtIndex(flatIndices[i], i * depth, depth, out, onValue)
    }
    return Tensor.new(out.toUIntArray(), Shape(targetShapeDims), indices.device())
}

/**
 * One-hot/cold encoding for 64-bit integer values.
 */
public fun oneHot(
    indices: Tensor,
    depth: Int,
    onValue: Long,
    offValue: Long,
): Tensor {
    val targetShapeDims = indices.dims().toMutableList()
    targetShapeDims.add(depth)
    val flatIndices = extractLongIndices(indices)
    val out = MutableList(depth * flatIndices.size) { offValue }
    for (i in flatIndices.indices) {
        setAtIndex(flatIndices[i], i * depth, depth, out, onValue)
    }
    return Tensor.new(out.toLongArray(), Shape(targetShapeDims), indices.device())
}
