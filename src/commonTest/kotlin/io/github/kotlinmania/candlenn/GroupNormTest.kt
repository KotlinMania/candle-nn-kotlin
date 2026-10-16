// port-lint: tests tests/group_norm.rs
package io.github.kotlinmania.candlenn

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.toVec3Round
import kotlin.test.Test
import kotlin.test.assertEquals

class GroupNormTest {
    @Test
    fun groupNorm() {
        val device = Device.Cpu
        val w = Tensor.fromVec(List(6) { 1.0f }, Shape(6), device)
        val b = Tensor.fromVec(List(6) { 0.0f }, Shape(6), device)
        val gn2 = GroupNorm.new(w, b, 6, 2, 1e-5)
        val gn3 = GroupNorm.new(w, b, 6, 3, 1e-5)

        val input =
            Tensor.new(
                floatArrayOf(
                    -0.3034f,
                    0.2726f,
                    -0.9659f,
                    -1.1845f,
                    -1.3236f,
                    0.0172f,
                    1.9507f,
                    1.2554f,
                    -0.8625f,
                    1.0682f,
                    0.3604f,
                    0.3985f,
                    -0.4957f,
                    -0.4461f,
                    -0.9721f,
                    1.5157f,
                    -0.1546f,
                    -0.5596f,
                    -1.6698f,
                    -0.4040f,
                    -0.7927f,
                    0.3736f,
                    -0.0975f,
                    -0.1351f,
                    -0.9461f,
                    0.5461f,
                    -0.6334f,
                    -1.0919f,
                    -0.1158f,
                    0.1213f,
                    -0.9535f,
                    0.1281f,
                    0.4372f,
                    -0.2845f,
                    0.3488f,
                    0.5641f,
                ),
                Shape(2, 6, 3),
                device,
            )
        assertEquals(
            listOf(
                listOf(
                    listOf(-0.1653f, 0.3748f, -0.7866f),
                    listOf(-0.9916f, -1.1220f, 0.1353f),
                    listOf(1.9485f, 1.2965f, -0.6896f),
                    listOf(1.2769f, 0.3628f, 0.4120f),
                    listOf(-0.7427f, -0.6786f, -1.3578f),
                    listOf(1.8547f, -0.3022f, -0.8252f),
                ),
                listOf(
                    listOf(-1.9342f, 0.0211f, -0.5793f),
                    listOf(1.2223f, 0.4945f, 0.4365f),
                    listOf(-0.8163f, 1.4887f, -0.3333f),
                    listOf(-1.7960f, -0.0392f, 0.3875f),
                    listOf(-1.5469f, 0.3998f, 0.9561f),
                    listOf(-0.3428f, 0.7970f, 1.1845f),
                ),
            ),
            toVec3Round(gn2.forward(input), 4),
        )
        assertEquals(
            listOf(
                listOf(
                    listOf(0.4560f, 1.4014f, -0.6313f),
                    listOf(-0.9901f, -1.2184f, 0.9822f),
                    listOf(1.4254f, 0.6360f, -1.7682f),
                    listOf(0.4235f, -0.3800f, -0.3367f),
                    listOf(-0.3890f, -0.3268f, -0.9862f),
                    listOf(2.1325f, 0.0386f, -0.4691f),
                ),
                listOf(
                    listOf(-1.8797f, 0.0777f, -0.5234f),
                    listOf(1.2802f, 0.5517f, 0.4935f),
                    listOf(-1.0102f, 1.5327f, -0.4773f),
                    listOf(-1.2587f, 0.4047f, 0.8088f),
                    listOf(-1.9074f, 0.1691f, 0.7625f),
                    listOf(-0.6230f, 0.5928f, 1.0061f),
                ),
            ),
            toVec3Round(gn3.forward(input), 4),
        )
    }
}
