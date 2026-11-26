package com.example.payscandemo

import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** A rectangle in PreviewView pixels (the on-screen viewfinder box). */
class RoiRect(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * Cuts the viewfinder box out of a camera frame and returns it as an upright grayscale strip.
 * Pure Kotlin (java.nio only), so the geometry can be unit-tested on the JVM.
 *
 * It undoes the same FILL_CENTER mapping that OcrAnalyzer.mapToView applies, in reverse:
 * view pixels -> upright image pixels -> sensor pixels (undoing rotationDegrees).
 */
object StripExtractor {
    /**
     * @param y             luminance plane: imageProxy.planes[0].buffer
     * @param rowStride     imageProxy.planes[0].rowStride (can be larger than the width!)
     * @param pixelStride   imageProxy.planes[0].pixelStride (1 for the Y plane on nearly all devices)
     * @param sensorW       imageProxy.width, before rotation
     * @param sensorH       imageProxy.height, before rotation
     * @param rotationDegrees imageProxy.imageInfo.rotationDegrees
     * @param outHeight     strip height in pixels (96 is a good size: sharp enough, cheap to process)
     */
    fun extract(
        y: ByteBuffer, rowStride: Int, pixelStride: Int,
        sensorW: Int, sensorH: Int, rotationDegrees: Int,
        viewW: Int, viewH: Int, roi: RoiRect, outHeight: Int = 96,
    ): GrayStrip? {
        if (viewW <= 0 || viewH <= 0) return null
        val rot = ((rotationDegrees % 360) + 360) % 360
        val upW = if (rot % 180 == 0) sensorW else sensorH
        val upH = if (rot % 180 == 0) sensorH else sensorW

        val scale = max(viewW / upW.toFloat(), viewH / upH.toFloat())
        val dx = (viewW - upW * scale) / 2f
        val dy = (viewH - upH * scale) / 2f
        val u0 = ((roi.left - dx) / scale).coerceIn(0f, upW.toFloat())
        val u1 = ((roi.right - dx) / scale).coerceIn(0f, upW.toFloat())
        val v0 = ((roi.top - dy) / scale).coerceIn(0f, upH.toFloat())
        val v1 = ((roi.bottom - dy) / scale).coerceIn(0f, upH.toFloat())
        if (u1 - u0 < 8f || v1 - v0 < 8f) return null

        val outW = (outHeight * (u1 - u0) / (v1 - v0)).roundToInt().coerceIn(16, 1600)
        val out = ByteArray(outW * outHeight)
        for (j in 0 until outHeight) {
            val uy = v0 + (j + 0.5f) * (v1 - v0) / outHeight - 0.5f
            for (i in 0 until outW) {
                val ux = u0 + (i + 0.5f) * (u1 - u0) / outW - 0.5f
                val sx: Float
                val sy: Float
                when (rot) {
                    90 -> { sx = uy; sy = sensorH - 1 - ux }
                    180 -> { sx = sensorW - 1 - ux; sy = sensorH - 1 - uy }
                    270 -> { sx = sensorW - 1 - uy; sy = ux }
                    else -> { sx = ux; sy = uy }
                }
                out[j * outW + i] = sample(y, rowStride, pixelStride, sensorW, sensorH, sx, sy).toByte()
            }
        }
        return GrayStrip(outW, outHeight, out)
    }

    private fun sample(y: ByteBuffer, rowStride: Int, pixelStride: Int, w: Int, h: Int, fx: Float, fy: Float): Int {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val yy = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = x.toInt()
        val y0 = yy.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = x - x0
        val ty = yy - y0
        fun px(cx: Int, cy: Int) = (y.get(cy * rowStride + cx * pixelStride).toInt() and 0xFF).toFloat()
        val top = px(x0, y0) * (1 - tx) + px(x1, y0) * tx
        val bottom = px(x0, y1) * (1 - tx) + px(x1, y1) * tx
        return (top * (1 - ty) + bottom * ty).roundToInt()
    }
}
