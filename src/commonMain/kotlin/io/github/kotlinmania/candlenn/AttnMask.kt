package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Tensor

public enum class AttnMaskKind {
    None,
    Causal,
    Mask,
}

public class AttnMask private constructor(
    public val kind: AttnMaskKind,
    private val causalKvOffset: Int = 0,
    public val maskTensor: Tensor? = null,
) {
    public constructor() : this(AttnMaskKind.None)

    public fun isCausal(): Boolean = kind == AttnMaskKind.Causal

    public fun kvOffset(): Int =
        when (kind) {
            AttnMaskKind.Causal -> causalKvOffset
            else -> 0
        }

    public companion object {
        public val None: AttnMask = AttnMask(AttnMaskKind.None)

        public fun Causal(kvOffset: Int = 0): AttnMask = AttnMask(AttnMaskKind.Causal, causalKvOffset = kvOffset)

        public fun Mask(tensor: Tensor): AttnMask = AttnMask(AttnMaskKind.Mask, maskTensor = tensor)

        public fun causal(): AttnMask = Causal(0)

        public fun causalWithOffset(kvOffset: Int): AttnMask = Causal(kvOffset)
    }
}
