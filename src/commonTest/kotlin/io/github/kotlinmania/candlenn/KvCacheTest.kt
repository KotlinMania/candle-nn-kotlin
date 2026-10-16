// port-lint: tests tests/kv_cache.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class KvCacheTest {
    @Test
    fun kvCache() {
        val cache = Cache.new(0, 16)
        for (iter in listOf(0, 1)) {
            assertEquals(0, cache.currentSeqLen())
            assertNull(cache.currentData())
            val t1 = Tensor.new(floatArrayOf(1.0f, 2.0f, 3.0f), Shape(3), Device.Cpu)
            cache.append(t1)
            assertEquals(listOf(1.0f, 2.0f, 3.0f), cache.currentData()!!.toVec1<Float>())
            val t2 = Tensor.new(floatArrayOf(4.0f), Shape(1), Device.Cpu)
            cache.append(t2)
            assertEquals(listOf(1.0f, 2.0f, 3.0f, 4.0f), cache.currentData()!!.toVec1<Float>())
            val t3 = Tensor.new(floatArrayOf(0.0f, 5.0f, 6.0f, 7.0f), Shape(4), Device.Cpu)
            cache.append(t3)
            assertEquals(
                listOf(1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 5.0f, 6.0f, 7.0f),
                cache.currentData()!!.toVec1<Float>(),
            )
            assertEquals(8, cache.currentSeqLen())
            cache.reset()
        }
    }

    @Test
    fun rotatingKvCache() {
        val cache = RotatingCache.new(0, 6)
        for (iter in listOf(0, 1)) {
            assertEquals(0, cache.offset())
            assertEquals(0, cache.currentSeqLen())
            assertNull(cache.currentData())
            assertEquals(listOf(0), cache.positions(1))
            assertEquals(listOf(0, 1), cache.positions(2))
            val t1 = Tensor.new(doubleArrayOf(1.0, 2.0, 3.0), Shape(3), Device.Cpu)
            val d1 = cache.append(t1)
            assertEquals(listOf(1.0, 2.0, 3.0), d1.toVec1<Double>())
            assertEquals(listOf(0, 1, 2), cache.positions(0))
            assertEquals(listOf(0, 1, 2, 3), cache.positions(1))
            assertEquals(listOf(0, 1, 2, 3, 4), cache.positions(2))
            assertEquals(listOf(0, 1, 2, 3, 4, 5), cache.positions(3))
            assertEquals(listOf(6, 1, 2, 3, 4, 5), cache.positions(4))
            val t2 = Tensor.new(doubleArrayOf(4.0), Shape(1), Device.Cpu)
            val d2 = cache.append(t2)
            assertEquals(listOf(1.0, 2.0, 3.0, 4.0), d2.toVec1<Double>())
            val t3 = Tensor.new(doubleArrayOf(0.0, 5.0, 6.0, 7.0), Shape(4), Device.Cpu)
            val d3 = cache.append(t3)
            assertEquals(listOf(6.0, 7.0, 3.0, 4.0, 0.0, 5.0), d3.toVec1<Double>())
            assertEquals(8, cache.currentSeqLen())
            assertEquals(2, cache.offset())

            val t4 = Tensor.new(doubleArrayOf(8.0), Shape(1), Device.Cpu)
            val d4 = cache.append(t4)
            assertEquals(listOf(6.0, 7.0, 8.0, 4.0, 0.0, 5.0), d4.toVec1<Double>())
            assertEquals(9, cache.currentSeqLen())
            assertEquals(3, cache.offset())

            val t5 = Tensor.new(doubleArrayOf(9.0, 10.0, 11.0), Shape(3), Device.Cpu)
            val d5 = cache.append(t5)
            assertEquals(listOf(6.0, 7.0, 8.0, 9.0, 10.0, 11.0), d5.toVec1<Double>())
            assertEquals(12, cache.currentSeqLen())
            assertEquals(0, cache.offset())

            val t6 = Tensor.new(doubleArrayOf(12.0), Shape(1), Device.Cpu)
            val d6 = cache.append(t6)
            assertEquals(listOf(12.0, 7.0, 8.0, 9.0, 10.0, 11.0), d6.toVec1<Double>())
            assertEquals(13, cache.currentSeqLen())
            assertEquals(1, cache.offset())

            val mask2 = cache.attnMask(2, Device.Cpu)!!
            assertEquals(
                listOf(
                    listOf<UByte>(0u, 0u, 1u, 0u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 0u, 0u, 0u, 0u),
                ),
                mask2.toVec2<UByte>(),
            )
            val mask3 = cache.attnMask(3, Device.Cpu)!!
            assertEquals(
                listOf(
                    listOf<UByte>(0u, 0u, 1u, 1u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 0u, 1u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 0u, 0u, 0u, 0u),
                ),
                mask3.toVec2<UByte>(),
            )
            assertEquals(listOf(12, 7, 8, 9, 10, 11), cache.positions(0))
            assertEquals(listOf(12, 13, 14, 9, 10, 11), cache.positions(2))
            assertEquals(listOf(12, 13, 14, 15, 10, 11), cache.positions(3))
            assertEquals(listOf(13, 14, 15, 16, 17, 18, 19, 20), cache.positions(8))

            val t7 = Tensor.new(doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), Shape(9), Device.Cpu)
            val d7 = cache.append(t7)
            assertEquals(listOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), d7.toVec1<Double>())
            assertEquals(22, cache.currentSeqLen())
            assertEquals(0, cache.offset())
            assertEquals(listOf(16, 17, 18, 19, 20, 21), cache.positions(0))
            assertEquals(listOf(22, 17, 18, 19, 20, 21), cache.positions(1))

            assertNull(cache.attnMask(1, Device.Cpu))
            val mask2b = cache.attnMask(2, Device.Cpu)!!
            assertEquals(
                listOf(
                    listOf<UByte>(0u, 1u, 0u, 0u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 0u, 0u, 0u, 0u),
                ),
                mask2b.toVec2<UByte>(),
            )
            val mask3b = cache.attnMask(3, Device.Cpu)!!
            assertEquals(
                listOf(
                    listOf<UByte>(0u, 1u, 1u, 0u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 1u, 0u, 0u, 0u),
                    listOf<UByte>(0u, 0u, 0u, 0u, 0u, 0u),
                ),
                mask3b.toVec2<UByte>(),
            )
            val t8 = Tensor.new(doubleArrayOf(42.0), Shape(1), Device.Cpu)
            val d8 = cache.append(t8)
            assertEquals(listOf(42.0, 4.0, 5.0, 6.0, 7.0, 8.0), d8.toVec1<Double>())
            assertEquals(23, cache.currentSeqLen())
            assertEquals(1, cache.offset())

            cache.reset()
        }
    }

    @Test
    fun testScatteredKvCache() {
        val device = Device.Cpu
        val cache = ScatteredCacheBuilder.new(2, 5, DType.F32, device)
        val inf = Float.POSITIVE_INFINITY

        val iam1 = cache.indicesAndMask(1, listOf(true, false))
        val mask1 = iam1.mask.narrow(1, 0, 1).squeeze(1).toVec3<Float>()
        assertEquals(listOf(listOf(0u), listOf(0u)), iam1.indices.toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(listOf(0.0f, -inf, -inf, -inf, -inf)),
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            ),
            mask1,
        )

        val iam2 = cache.indicesAndMask(1, listOf(true, false))
        val mask2 = iam2.mask.narrow(1, 0, 1).squeeze(1).toVec3<Float>()
        assertEquals(listOf(listOf(1u), listOf(0u)), iam2.indices.toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(listOf(0.0f, 0.0f, -inf, -inf, -inf)),
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            ),
            mask2,
        )

        val iam3 = cache.indicesAndMask(3, listOf(false, true))
        val mask3 = iam3.mask.narrow(1, 0, 1).squeeze(1).toVec3<Float>()
        assertEquals(listOf(listOf(2u, 2u, 2u), listOf(0u, 1u, 2u)), iam3.indices.toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
                listOf(
                    listOf(0.0f, -inf, -inf, -inf, -inf),
                    listOf(0.0f, 0.0f, -inf, -inf, -inf),
                    listOf(0.0f, 0.0f, 0.0f, -inf, -inf),
                ),
            ),
            mask3,
        )
    }

    @Test
    fun testConcatCacheBasic() {
        val device = Device.Cpu
        val cache = ConcatKvCache.new(2)

        assertTrue(cache.isEmpty())
        assertEquals(0, cache.currentSeqLen())

        val k1 = Tensor.zeros(Shape(1, 8, 3, 64), DType.F32, device)
        val v1 = Tensor.zeros(Shape(1, 8, 3, 64), DType.F32, device)
        val (k, v) = cache.append(k1, v1)

        assertEquals(listOf(1, 8, 3, 64), k.dims())
        assertEquals(listOf(1, 8, 3, 64), v.dims())
        assertEquals(3, cache.currentSeqLen())
        assertFalse(cache.isEmpty())

        val k2 = Tensor.zeros(Shape(1, 8, 2, 64), DType.F32, device)
        val v2 = Tensor.zeros(Shape(1, 8, 2, 64), DType.F32, device)
        val (kNext, vNext) = cache.append(k2, v2)

        assertEquals(listOf(1, 8, 5, 64), kNext.dims())
        assertEquals(listOf(1, 8, 5, 64), vNext.dims())
        assertEquals(5, cache.currentSeqLen())
    }

    @Test
    fun testConcatCacheReset() {
        val device = Device.Cpu
        val cache = ConcatKvCache.new(2)

        val k = Tensor.zeros(Shape(1, 8, 10, 64), DType.F32, device)
        val v = Tensor.zeros(Shape(1, 8, 10, 64), DType.F32, device)
        cache.append(k, v)
        assertEquals(10, cache.currentSeqLen())

        cache.reset()
        assertTrue(cache.isEmpty())
        assertEquals(0, cache.currentSeqLen())
        assertNull(cache.k())
        assertNull(cache.v())
    }

    @Test
    fun testConcatCacheMultipleAppends() {
        val device = Device.Cpu
        val cache = ConcatKvCache.new(2)

        val kPrefill = Tensor.zeros(Shape(1, 8, 10, 64), DType.F32, device)
        val vPrefill = Tensor.zeros(Shape(1, 8, 10, 64), DType.F32, device)
        cache.append(kPrefill, vPrefill)
        assertEquals(10, cache.currentSeqLen())

        for (i in 1..5) {
            val kToken = Tensor.zeros(Shape(1, 8, 1, 64), DType.F32, device)
            val vToken = Tensor.zeros(Shape(1, 8, 1, 64), DType.F32, device)
            val (k, v) = cache.append(kToken, vToken)
            assertEquals(10 + i, k.dims()[2])
            assertEquals(10 + i, v.dims()[2])
        }
        assertEquals(15, cache.currentSeqLen())
    }

    @Test
    fun testConcatCacheDifferentDim() {
        val device = Device.Cpu
        val cache = ConcatKvCache.new(1)

        val k1 = Tensor.zeros(Shape(1, 3, 8, 64), DType.F32, device)
        val v1 = Tensor.zeros(Shape(1, 3, 8, 64), DType.F32, device)
        val (k, _) = cache.append(k1, v1)
        assertEquals(listOf(1, 3, 8, 64), k.dims())

        val k2 = Tensor.zeros(Shape(1, 2, 8, 64), DType.F32, device)
        val v2 = Tensor.zeros(Shape(1, 2, 8, 64), DType.F32, device)
        val (kNext, _) = cache.append(k2, v2)
        assertEquals(listOf(1, 5, 8, 64), kNext.dims())
        assertEquals(5, cache.currentSeqLen())
    }
}
