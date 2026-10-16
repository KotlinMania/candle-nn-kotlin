// port-lint: source var_map.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.MmapedSafetensors
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.Var
import io.github.kotlinmania.candle.save

/**
 * A [VarMap] is a store that holds named variables. Variables can be retrieved from the stores
 * and new variables can be added by providing some initialization config in case they are
 * missing.
 * [VarMap] structures can be serialized in the safetensors format.
 */
public class VarMap internal constructor(
    private val mapData: MutableMap<String, Var> = LinkedHashMap(),
) {
    /**
     * Retrieve all the variables currently stored in the map.
     */
    public fun allVars(): List<Var> = mapData.values.toList()

    /**
     * Save the map in the safetensors format.
     */
    public fun save(path: String) {
        val tensorMap = LinkedHashMap<String, Tensor>(mapData.size)
        for ((k, v) in mapData) {
            tensorMap[k] = v.asTensor()
        }
        save(tensorMap, path).getOrThrow()
    }

    /**
     * Load some values from a safetensors file and modify the existing variables to have these
     * values.
     *
     * Note that values for variables that are currently not in the map are not kept.
     */
    public fun load(path: String) {
        val loaded = MmapedSafetensors.new(path).getOrThrow()
        for ((name, variable) in mapData) {
            val tensor = loaded.load(name, variable.asTensor().device()).getOrThrow()
            try {
                variable.set(tensor)
            } catch (err: Throwable) {
                throw CandleException.Msg("error setting $name using data from $path: ${err.message}")
            }
        }
    }

    /**
     * Set a named variable to some value.
     */
    public fun setOne(
        name: String,
        value: Tensor,
    ) {
        val variable = mapData[name] ?: throw CandleException.Msg("cannot find $name in VarMap")
        try {
            variable.set(value)
        } catch (err: Throwable) {
            throw CandleException.Msg("error setting $name: ${err.message}")
        }
    }

    /**
     * Set some named variables to some values.
     *
     * If an error is returned, some of the variables might have already been set to their new
     * values.
     */
    public fun set(entries: Map<String, Tensor>) {
        for ((name, value) in entries) {
            val variable = mapData[name] ?: throw CandleException.Msg("cannot find $name in VarMap")
            try {
                variable.set(value)
            } catch (err: Throwable) {
                throw CandleException.Msg("error setting $name: ${err.message}")
            }
        }
    }

    /**
     * Retrieve or add a new variable.
     */
    public fun get(
        shape: Shape,
        path: String,
        init: Init,
        dtype: DType,
        device: Device,
    ): Tensor {
        val existing = mapData[path]
        if (existing != null) {
            val tensorShape = existing.asTensor().shape()
            if (shape != tensorShape) {
                throw CandleException.Msg("shape mismatch on $path: $shape <> $tensorShape")
            }
            return existing.asTensor()
        }
        val variable = init.createVar(shape, dtype, device)
        val tensor = variable.asTensor()
        mapData[path] = variable
        return tensor
    }

    public fun get(
        shape: Int,
        path: String,
        init: Init,
        dtype: DType,
        device: Device,
    ): Tensor = get(Shape(shape), path, init, dtype, device)

    public fun get(
        shape: Pair<Int, Int>,
        path: String,
        init: Init,
        dtype: DType,
        device: Device,
    ): Tensor = get(Shape(shape.first, shape.second), path, init, dtype, device)

    public fun get(
        shape: Triple<Int, Int, Int>,
        path: String,
        init: Init,
        dtype: DType,
        device: Device,
    ): Tensor = get(Shape(shape.first, shape.second, shape.third), path, init, dtype, device)

    public fun get(
        shape: List<Int>,
        path: String,
        init: Init,
        dtype: DType,
        device: Device,
    ): Tensor = get(Shape(shape), path, init, dtype, device)

    public fun data(): Map<String, Var> = mapData

    public fun clone(): VarMap = VarMap(mapData)

    public companion object {
        /**
         * Create a new empty [VarMap].
         */
        public fun new(): VarMap = VarMap(LinkedHashMap())
    }
}
