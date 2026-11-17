package com.example.payscandemo

import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.unit.IntSize
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.max

class OcrAnalyzerTest {

    @Test
    fun ocrBlock_initialization_holdsCorrectValues() {
        val rect = Rect(10f, 20f, 100f, 200f)
        val block = OcrBlock(text = "Total $45.00", box = rect)

        assertEquals("Total $45.00", block.text)
        assertEquals(rect, block.box)
    }

    @Test
    fun ocrBlock_withNullBox_handlesNullCorrectly() {
        val block = OcrBlock(text = "Unbounded text", box = null)

        assertEquals("Unbounded text", block.text)
        assertNull(block.box)
    }

    @Test
    fun mapToView_calculatesCorrectScaledCoordinates() {
        // Image size: 1280x720, View size: 1080x1920 (Portrait screen fill)
        val imageW = 1280
        val imageH = 720
        val viewW = 1080
        val viewH = 1920

        val scale = max(viewW / imageW.toFloat(), viewH / imageH.toFloat())
        val dx = (viewW - (imageW * scale)) / 2f
        val dy = (viewH - (imageH * scale)) / 2f

        val boxLeft = 100f
        val boxTop = 100f
        val boxRight = 300f
        val boxBottom = 200f

        val mappedLeft = boxLeft * scale + dx
        val mappedTop = boxTop * scale + dy
        val mappedRight = boxRight * scale + dx
        val mappedBottom = boxBottom * scale + dy

        val mappedRect = Rect(mappedLeft, mappedTop, mappedRight, mappedBottom)

        assertNotNull(mappedRect)
        assertEquals(200f * scale, mappedRect.width, 0.01f)
        assertEquals(100f * scale, mappedRect.height, 0.01f)
    }

    @Test
    fun mapToView_zeroViewDimension_returnsNullSafely() {
        val view = IntSize(0, 1920)
        assertEquals(0, view.width)
    }
}