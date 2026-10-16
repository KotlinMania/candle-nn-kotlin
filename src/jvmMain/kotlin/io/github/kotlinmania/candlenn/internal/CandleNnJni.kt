// port-lint: source N/A
package io.github.kotlinmania.candlenn.internal

internal object CandleNnJni {
    init {
        System.loadLibrary("candlenn_jni")
    }

    @JvmStatic
    external fun getpid(): Int
}
