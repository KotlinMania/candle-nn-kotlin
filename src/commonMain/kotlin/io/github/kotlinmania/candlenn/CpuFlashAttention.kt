// port-lint: source cpu_flash_attention.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Tensor

/**
 * Backward compatibility shim for CPU flash attention.
 *
 * Routes through the attention dispatcher which handles both B=1 and B>1.
 */
public fun runFlashAttnCpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor?,
    softmaxScale: Float,
    maxBias: Float? = null,
    softcap: Float? = null,
): Tensor {
    val attnMask =
        if (mask != null) {
            AttnMask.Mask(mask)
        } else {
            AttnMask.None
        }
    return flashAttn(q, k, v, softmaxScale, attnMask, maxBias, softcap)
}
