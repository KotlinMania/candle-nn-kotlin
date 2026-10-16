// port-lint: source rnn.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.cat

/**
 * The state for a LSTM network, this contains two tensors.
 */
public class LSTMState(
    internal val h: Tensor,
    internal val c: Tensor,
) {
    public companion object {
        public fun new(h: Tensor, c: Tensor): LSTMState = LSTMState(h, c)
    }

    /**
     * The hidden state vector, which is also the output of the LSTM.
     */
    public fun h(): Tensor = h

    /**
     * The cell state vector.
     */
    public fun c(): Tensor = c

    public fun clone(): LSTMState = LSTMState(h, c)
}

public enum class Direction {
    Forward,
    Backward,
}

public data class LSTMConfig(
    public val wIhInit: Init = DEFAULT_KAIMING_UNIFORM,
    public val wHhInit: Init = DEFAULT_KAIMING_UNIFORM,
    public val bIhInit: Init? = Init.Const(0.0),
    public val bHhInit: Init? = Init.Const(0.0),
    public val layerIdx: Int = 0,
    public val direction: Direction = Direction.Forward,
) {
    public companion object {
        public fun default(): LSTMConfig = LSTMConfig()

        public fun defaultNoBias(): LSTMConfig = LSTMConfig(
            wIhInit = DEFAULT_KAIMING_UNIFORM,
            wHhInit = DEFAULT_KAIMING_UNIFORM,
            bIhInit = null,
            bHhInit = null,
            layerIdx = 0,
            direction = Direction.Forward,
        )
    }
}

/**
 * A Long Short-Term Memory (LSTM) layer.
 */
public class LSTM(
    private val wIh: Tensor,
    private val wHh: Tensor,
    private val bIh: Tensor?,
    private val bHh: Tensor?,
    private val hiddenDim: Int,
    private val config: LSTMConfig,
    private val device: Device,
    private val dtype: DType,
) {
    public companion object {
        /**
         * Creates a LSTM layer.
         */
        public fun new(
            inDim: Int,
            hiddenDim: Int,
            config: LSTMConfig,
            vb: VarBuilder,
        ): LSTM {
            val layerIdx = config.layerIdx
            val directionStr = when (config.direction) {
                Direction.Forward -> ""
                Direction.Backward -> "_reverse"
            }
            val wIh = vb.getWithHints(
                Shape(4 * hiddenDim, inDim),
                "weight_ih_l$layerIdx$directionStr",
                config.wIhInit,
            )
            val wHh = vb.getWithHints(
                Shape(4 * hiddenDim, hiddenDim),
                "weight_hh_l$layerIdx$directionStr",
                config.wHhInit,
            )
            val bIh = config.bIhInit?.let { init ->
                vb.getWithHints(
                    4 * hiddenDim,
                    "bias_ih_l$layerIdx$directionStr",
                    init,
                )
            }
            val bHh = config.bHhInit?.let { init ->
                vb.getWithHints(
                    4 * hiddenDim,
                    "bias_hh_l$layerIdx$directionStr",
                    init,
                )
            }
            return LSTM(
                wIh = wIh,
                wHh = wHh,
                bIh = bIh,
                bHh = bHh,
                hiddenDim = hiddenDim,
                config = config,
                device = vb.device(),
                dtype = vb.dtype(),
            )
        }
    }

    public fun config(): LSTMConfig = config

    public fun zeroState(batchDim: Int): LSTMState {
        val zeros = Tensor.zeros(Shape(batchDim, hiddenDim), dtype, device).contiguous()
        return LSTMState(h = zeros, c = zeros)
    }

    public fun step(input: Tensor, inState: LSTMState): LSTMState {
        val wIhOut0 = input.matmul(wIh.t())
        val wHhOut0 = inState.h.matmul(wHh.t())
        val wIhOut = when (bIh) {
            null -> wIhOut0
            else -> wIhOut0.broadcastAdd(bIh)
        }
        val wHhOut = when (bHh) {
            null -> wHhOut0
            else -> wHhOut0.broadcastAdd(bHh)
        }
        val chunks = (wIhOut + wHhOut).chunk(4, 1)
        val inGate = sigmoid(chunks[0])
        val forgetGate = sigmoid(chunks[1])
        val cellGate = chunks[2].tanh()
        val outGate = sigmoid(chunks[3])

        val nextC = (forgetGate * inState.c) + (inGate * cellGate)
        val nextH = outGate * nextC.tanh()
        return LSTMState(c = nextC, h = nextH)
    }

    public fun seq(input: Tensor): List<LSTMState> {
        val batchDim = input.dim(0)
        val state = zeroState(batchDim)
        return seqInit(input, state)
    }

    public fun seqInit(input: Tensor, initState: LSTMState): List<LSTMState> {
        val (_, seqLen, _) = input.dims3()
        val output = ArrayList<LSTMState>(seqLen)
        for (seqIndex in 0 until seqLen) {
            val stepInput = input.narrow(1, seqIndex, 1).squeeze(1).contiguous()
            val state = if (seqIndex == 0) {
                step(stepInput, initState)
            } else {
                step(stepInput, output[seqIndex - 1])
            }
            output.add(state)
        }
        return output
    }

    public fun statesToTensor(states: List<LSTMState>): Tensor {
        val hTensors = states.map { it.h }
        return Tensor.stack(hTensors, 1)
    }
}

/**
 * Creates a LSTM layer.
 */
public fun lstm(
    inDim: Int,
    hiddenDim: Int,
    config: LSTMConfig,
    vb: VarBuilder,
): LSTM = LSTM.new(inDim, hiddenDim, config, vb)

/**
 * The state for a GRU network, this contains a single tensor.
 */
public class GRUState(
    internal val h: Tensor,
) {
    /**
     * The hidden state vector, which is also the output of the GRU.
     */
    public fun h(): Tensor = h

    public fun clone(): GRUState = GRUState(h)
}

public data class GRUConfig(
    public val wIhInit: Init = DEFAULT_KAIMING_UNIFORM,
    public val wHhInit: Init = DEFAULT_KAIMING_UNIFORM,
    public val bIhInit: Init? = Init.Const(0.0),
    public val bHhInit: Init? = Init.Const(0.0),
) {
    public companion object {
        public fun default(): GRUConfig = GRUConfig()

        public fun defaultNoBias(): GRUConfig = GRUConfig(
            wIhInit = DEFAULT_KAIMING_UNIFORM,
            wHhInit = DEFAULT_KAIMING_UNIFORM,
            bIhInit = null,
            bHhInit = null,
        )
    }
}

/**
 * A Gated Recurrent Unit (GRU) layer.
 */
public class GRU(
    private val wIh: Tensor,
    private val wHh: Tensor,
    private val bIh: Tensor?,
    private val bHh: Tensor?,
    private val hiddenDim: Int,
    private val config: GRUConfig,
    private val device: Device,
    private val dtype: DType,
) {
    public companion object {
        /**
         * Creates a GRU layer.
         */
        public fun new(
            inDim: Int,
            hiddenDim: Int,
            config: GRUConfig,
            vb: VarBuilder,
        ): GRU {
            val wIh = vb.getWithHints(
                Shape(3 * hiddenDim, inDim),
                "weight_ih_l0",
                config.wIhInit,
            )
            val wHh = vb.getWithHints(
                Shape(3 * hiddenDim, hiddenDim),
                "weight_hh_l0",
                config.wHhInit,
            )
            val bIh = config.bIhInit?.let { init ->
                vb.getWithHints(3 * hiddenDim, "bias_ih_l0", init)
            }
            val bHh = config.bHhInit?.let { init ->
                vb.getWithHints(3 * hiddenDim, "bias_hh_l0", init)
            }
            return GRU(
                wIh = wIh,
                wHh = wHh,
                bIh = bIh,
                bHh = bHh,
                hiddenDim = hiddenDim,
                config = config,
                device = vb.device(),
                dtype = vb.dtype(),
            )
        }
    }

    public fun config(): GRUConfig = config

    public fun zeroState(batchDim: Int): GRUState {
        val h = Tensor.zeros(Shape(batchDim, hiddenDim), dtype, device).contiguous()
        return GRUState(h)
    }

    public fun step(input: Tensor, inState: GRUState): GRUState {
        val wIhOut0 = input.matmul(wIh.t())
        val wHhOut0 = inState.h.matmul(wHh.t())
        val wIhOut = when (bIh) {
            null -> wIhOut0
            else -> wIhOut0.broadcastAdd(bIh)
        }
        val wHhOut = when (bHh) {
            null -> wHhOut0
            else -> wHhOut0.broadcastAdd(bHh)
        }
        val chunksIh = wIhOut.chunk(3, 1)
        val chunksHh = wHhOut.chunk(3, 1)
        val rGate = sigmoid(chunksIh[0] + chunksHh[0])
        val zGate = sigmoid(chunksIh[1] + chunksHh[1])
        val nGate = (chunksIh[2] + (rGate * chunksHh[2])).tanh()

        val nextH = (zGate * inState.h) - ((zGate - 1.0) * nGate)
        return GRUState(h = nextH)
    }

    public fun seq(input: Tensor): List<GRUState> {
        val batchDim = input.dim(0)
        val state = zeroState(batchDim)
        return seqInit(input, state)
    }

    public fun seqInit(input: Tensor, initState: GRUState): List<GRUState> {
        val (_, seqLen, _) = input.dims3()
        val output = ArrayList<GRUState>(seqLen)
        for (seqIndex in 0 until seqLen) {
            val stepInput = input.narrow(1, seqIndex, 1).squeeze(1).contiguous()
            val state = if (seqIndex == 0) {
                step(stepInput, initState)
            } else {
                step(stepInput, output[seqIndex - 1])
            }
            output.add(state)
        }
        return output
    }

    public fun statesToTensor(states: List<GRUState>): Tensor {
        val hTensors = states.map { it.h }
        return Tensor.cat(hTensors, 1)
    }
}

public fun gru(
    inDim: Int,
    hiddenDim: Int,
    config: GRUConfig,
    vb: VarBuilder,
): GRU = GRU.new(inDim, hiddenDim, config, vb)
