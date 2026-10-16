// port-lint: tests tests/one_hot.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.test.Test
import kotlin.test.assertEquals

class OneHotTest {
    @Test
    fun testI64OneHot() {
        val device = Device.Cpu
        val indices = Tensor.new(longArrayOf(0L, 2L, 1L, -1L), Shape(2, 2), device)
        val depth = 4
        val oh = oneHot(indices, depth, 1.0f, 0.0f)

        val expectedMatrix = listOf(
            listOf(listOf(1.0f, 0.0f, 0.0f, 0.0f), listOf(0.0f, 0.0f, 1.0f, 0.0f)),
            listOf(listOf(0.0f, 1.0f, 0.0f, 0.0f), listOf(0.0f, 0.0f, 0.0f, 0.0f)),
        )
        assertEquals(Shape(2, 2, depth), oh.shape())
        assertEquals(expectedMatrix, oh.toVec3<Float>())
    }

    @Test
    fun testRank3OneHot() {
        val device = Device.Cpu
        val indices = Tensor.new(
            longArrayOf(0L, 1L, 2L, 3L, 3L, 1L, 1L, -1L),
            Shape(2, 2, 2),
            device,
        )
        val depth = 4
        val oh = oneHot(indices, depth, 1.0f, 0.0f)

        val expectedMatrix = Tensor.fromVec(
            listOf(
                1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            ),
            Shape(2, 2, 2, 4),
            device,
        )

        assertEquals(expectedMatrix.shape(), oh.shape())
        assertEquals(expectedMatrix.dims(), oh.dims())
        assertEquals(expectedMatrix.get(1).toVec3<Float>(), oh.get(1).toVec3<Float>())
    }

    @Test
    fun testU8OneCold() {
        val device = Device.Cpu
        val depth = 4
        val indices = Tensor.new(longArrayOf(0L, 2L, 1L, -1L), Shape(2, 2), device)
        val oneCold = oneHot(indices, depth, 0u.toUByte(), 1u.toUByte())

        val expectedMatrix = listOf(
            listOf(listOf<UByte>(0u, 1u, 1u, 1u), listOf<UByte>(1u, 1u, 0u, 1u)),
            listOf(listOf<UByte>(1u, 0u, 1u, 1u), listOf<UByte>(1u, 1u, 1u, 1u)),
        )
        assertEquals(Shape(2, 2, depth), oneCold.shape())
        assertEquals(expectedMatrix, oneCold.toVec3<UByte>())
    }

    @Test
    fun testIter() {
        val device = Device.Cpu
        val depth = 4
        val indices = Tensor.new(longArrayOf(0L, 2L, 1L, -1L), Shape(2, 2), device)
        val matrix = indices.toVec2<Long>()
        val (dim1, dim2) = indices.dims2()

        val v = LongArray(depth * dim1 * dim2)
        for (i in 0 until dim1) {
            for (j in 0 until dim2) {
                val idx = i * depth * dim2 + j * depth
                v[idx] = matrix[i][j]
            }
        }

        for ((i, row) in matrix.withIndex()) {
            for ((j, value) in row.withIndex()) {
                val idx = i * depth * dim2 + j * depth
                assertEquals(value, v[idx])
            }
        }
    }
}
