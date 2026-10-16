// port-lint: source attention/cpu_flash/online_softmax.rs
package io.github.kotlinmania.candlenn

import kotlin.math.exp

/**
 * Stream one (score, vRow) into (m, ssum, acc); vApply adds vRow * w to acc.
 */
internal inline fun onlineSoftmaxStep(
    score: Float,
    m: FloatArray,
    ssum: FloatArray,
    acc: FloatArray,
    vApply: (FloatArray, Float) -> Unit,
) {
    val curM = m[0]
    if (score > curM) {
        val scaleOld = exp(curM - score)
        for (i in acc.indices) {
            acc[i] *= scaleOld
        }
        ssum[0] = ssum[0] * scaleOld + 1.0f
        m[0] = score
        vApply(acc, 1.0f)
    } else {
        val w = exp(score - curM)
        vApply(acc, w)
        ssum[0] += w
    }
}
