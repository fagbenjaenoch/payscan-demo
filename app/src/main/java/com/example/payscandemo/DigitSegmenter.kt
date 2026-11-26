package com.example.payscandemo

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/*
 * Cuts a strip of handwriting into single-digit 32x32 crops. Pure Kotlin (no Android classes),
 * so it can be unit-tested on the JVM. Expects DARK ink on LIGHT paper.
 *
 * The crop format must match train/digit_utils.py: 32x32, ink = 1.0, background = 0.0, the digit
 * scaled to fit 24x24 (4 px padding) keeping its aspect ratio and centred.
 *
 * Steps: 1) adaptive threshold  2) erase long horizontal runs (ruled lines)  3) connected components
 *        4) attach small fragments (a detached 5 bar), merge broken digits  5) adjust to the expected
 *        count by splitting touching digits or merging stray pieces  6) normalise each digit.
 */

/** Grayscale image, row-major, 0 = black .. 255 = white. */
class GrayStrip(val width: Int, val height: Int, val pixels: ByteArray) {
    init {
        require(pixels.size == width * height) { "pixels must hold width*height values" }
    }

    fun at(x: Int, y: Int): Int = pixels[y * width + x].toInt() and 0xFF
}

/** Inclusive pixel box inside the strip. */
class DigitBox(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = right - left + 1
    val height get() = bottom - top + 1
}

class Segmentation(
    /** One 32x32 FloatArray (row-major, 0..1, ink = 1) per digit, left to right. */
    val crops: List<FloatArray>,
    val boxes: List<DigitBox>,
    /** Why the result may be unusable (null when nothing odd happened). */
    val note: String? = null,
)

object DigitSegmenter {
    const val SIZE = 32
    private const val PAD = 4
    private const val MIN_CONTRAST = 14

    fun segment(strip: GrayStrip, expected: IntRange = 10..11): Segmentation {
        val w = strip.width
        val h = strip.height
        if (w < 20 || h < 20) return Segmentation(emptyList(), emptyList(), "strip too small")

        val ink = binarize(strip)
        if (ink.count { it } > ink.size * 0.4) return Segmentation(emptyList(), emptyList(), "image too dark")
        removeLongHorizontalRuns(ink, w, h)

        val minPixels = max(6, h * h / 500)
        var blobs = findBlobs(ink, w, h).filter { it.size >= minPixels }

        // Reference digit height = median height of tall, not-too-wide blobs.
        val tall = blobs.filter { it.height >= 0.3 * h && it.width <= 1.8 * it.height }
        if (tall.isEmpty()) return Segmentation(emptyList(), emptyList(), "no digit-sized shapes found")
        val refH = median(tall.map { it.height.toDouble() })

        // Attach fragments (small or flat pieces) to the main blob they sit over; drop the rest.
        val main = blobs.filter { it.height >= 0.5 * refH && it.width <= 2.6 * it.height }.toMutableList()
        for (f in blobs) {
            if (f in main || f.width > 1.3 * refH) continue
            val target = main.maxByOrNull { overlap(it, f) } ?: continue
            if (overlap(target, f) >= 0.4 * f.width && unionHeight(target, f) <= 1.35 * refH) target.absorb(f)
        }
        blobs = main.sortedBy { it.left }

        blobs = mergeBrokenDigits(blobs, refH)
        blobs = adjustCount(blobs, refH, expected)

        val note = if (blobs.size in expected) null else "found ${blobs.size} digit shapes, expected $expected"
        return Segmentation(
            crops = blobs.map { toCrop(it) },
            boxes = blobs.map { DigitBox(it.left, it.top, it.right, it.bottom) },
            note = note,
        )
    }

    // ------------------------------------------------------------------ 1) threshold
    /**
     * Washed-out or blurry photos use only a narrow band of grey levels (e.g. 150..205), so a fixed
     * "14 levels darker" rule either finds nothing or fills everything. Rescaling the 2nd..98th
     * percentile of brightness to 0..255 first makes the rule mean the same thing for every photo.
     */
    private fun stretchContrast(s: GrayStrip): IntArray {
        val n = s.width * s.height
        val hist = IntArray(256)
        for (i in 0 until n) hist[s.pixels[i].toInt() and 0xFF]++
        fun percentile(p: Int): Int {
            val target = n.toLong() * p / 100
            var acc = 0L
            for (v in 0..255) { acc += hist[v]; if (acc > target) return v }
            return 255
        }
        val lo = percentile(2)
        val hi = percentile(98)
        val out = IntArray(n)
        val range = hi - lo
        if (range < 8) {                       // flat picture: leave it alone, nothing to stretch
            for (i in 0 until n) out[i] = s.pixels[i].toInt() and 0xFF
            return out
        }
        for (i in 0 until n) out[i] = ((((s.pixels[i].toInt() and 0xFF) - lo) * 255) / range).coerceIn(0, 255)
        return out
    }

    /** Bradley-Roth style: a pixel is ink if it is clearly darker than the average of its neighbourhood. */
    private fun binarize(s: GrayStrip): BooleanArray {
        val w = s.width
        val h = s.height
        val g = stretchContrast(s)
        val stride = w + 1
        val integral = IntArray(stride * (h + 1))
        for (y in 0 until h) {
            var row = 0
            for (x in 0 until w) {
                row += g[y * w + x]
                integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row
            }
        }
        val half = max(7, h / 8)
        val ink = BooleanArray(w * h)
        for (y in 0 until h) {
            val y0 = max(0, y - half)
            val y1 = min(h - 1, y + half)
            for (x in 0 until w) {
                val x0 = max(0, x - half)
                val x1 = min(w - 1, x + half)
                val area = (x1 - x0 + 1) * (y1 - y0 + 1)
                val sum = integral[(y1 + 1) * stride + x1 + 1] - integral[y0 * stride + x1 + 1] -
                    integral[(y1 + 1) * stride + x0] + integral[y0 * stride + x0]
                val mean = sum / area
                ink[y * w + x] = mean - g[y * w + x] >= max(MIN_CONTRAST, mean * 15 / 100)
            }
        }
        return ink
    }

    // ------------------------------------------------------------------ 2) ruled lines
    private fun removeLongHorizontalRuns(ink: BooleanArray, w: Int, h: Int) {
        val limit = (1.2 * h).toInt()
        for (y in 0 until h) {
            var x = 0
            while (x < w) {
                if (!ink[y * w + x]) { x++; continue }
                var e = x
                while (e < w && ink[y * w + e]) e++
                if (e - x > limit) for (i in x until e) ink[y * w + i] = false
                x = e
            }
        }
    }

    // ------------------------------------------------------------------ 3) components
    private class IntList {
        var a = IntArray(64)
        var size = 0
        fun add(v: Int) { if (size == a.size) a = a.copyOf(size * 2); a[size++] = v }
    }

    private class Blob(val stripW: Int) {
        val px = IntList()
        var left = Int.MAX_VALUE
        var top = Int.MAX_VALUE
        var right = -1
        var bottom = -1
        val size get() = px.size
        val width get() = right - left + 1
        val height get() = bottom - top + 1

        fun add(index: Int) {
            px.add(index)
            val x = index % stripW
            val y = index / stripW
            if (x < left) left = x
            if (x > right) right = x
            if (y < top) top = y
            if (y > bottom) bottom = y
        }

        fun absorb(o: Blob) { for (i in 0 until o.px.size) add(o.px.a[i]) }
    }

    /** Clears [ink] as it goes. 8-connected flood fill with an explicit stack. */
    private fun findBlobs(ink: BooleanArray, w: Int, h: Int): List<Blob> {
        val out = ArrayList<Blob>()
        val stack = IntArray(w * h)
        for (start in ink.indices) {
            if (!ink[start]) continue
            val b = Blob(w)
            var sp = 0
            stack[sp++] = start
            ink[start] = false
            while (sp > 0) {
                val p = stack[--sp]
                b.add(p)
                val x = p % w
                val y = p / w
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
                    val n = ny * w + nx
                    if (ink[n]) { ink[n] = false; stack[sp++] = n }
                }
            }
            out += b
        }
        return out
    }

    // ------------------------------------------------------------------ 4) merging
    private fun overlap(a: Blob, b: Blob): Int = max(0, min(a.right, b.right) - max(a.left, b.left) + 1)

    private fun unionHeight(a: Blob, b: Blob): Int = max(a.bottom, b.bottom) - min(a.top, b.top) + 1

    /** Pieces of one digit that sit mostly on top of each other (a 4 or 5 drawn in two strokes). */
    private fun mergeBrokenDigits(sorted: List<Blob>, refH: Double): List<Blob> {
        val list = sorted.toMutableList()
        var i = 0
        while (i < list.size - 1) {
            val a = list[i]
            val b = list[i + 1]
            val mergedWidth = max(a.right, b.right) - min(a.left, b.left) + 1
            if (overlap(a, b) >= 0.7 * min(a.width, b.width) && mergedWidth <= 1.0 * refH) {
                a.absorb(b)
                list.removeAt(i + 1)
            } else i++
        }
        return list
    }

    // ------------------------------------------------------------------ 5) adjust the count
    private fun adjustCount(input: List<Blob>, refH: Double, expected: IntRange): List<Blob> {
        val list = input.sortedBy { it.left }.toMutableList()

        // Too many: join the closest pair, but only if they are really close (stray pieces).
        while (list.size > expected.last) {
            var best = -1
            var bestGap = Int.MAX_VALUE
            for (i in 0 until list.size - 1) {
                val merged = max(list[i].right, list[i + 1].right) - min(list[i].left, list[i + 1].left) + 1
                val gap = list[i + 1].left - list[i].right - 1
                if (merged <= 1.15 * refH && gap < bestGap) { bestGap = gap; best = i }
            }
            if (best < 0 || bestGap > 0.25 * refH) break
            list[best].absorb(list[best + 1])
            list.removeAt(best + 1)
        }

        // Too few: split the widest blob that is wide enough to hold two digits.
        while (list.size < expected.first) {
            val widths = list.map { it.width.toDouble() }
            var pick = -1
            var pickW = 0
            for (i in list.indices) {
                val others = widths.filterIndexed { j, _ -> j != i }
                val med = if (others.size >= 3) median(others) else 0.55 * refH
                val b = list[i]
                if ((b.width >= 0.85 * refH || b.width >= 1.6 * med) && b.width > pickW) { pick = i; pickW = b.width }
            }
            if (pick < 0) break
            val parts = split(list[pick])
            list.removeAt(pick)
            list.addAll(pick, parts)
        }
        return list
    }

    /** Cuts a blob in two at the column (near its middle) that crosses the least ink. */
    private fun split(b: Blob): List<Blob> {
        val cols = IntArray(b.width)
        for (i in 0 until b.px.size) cols[b.px.a[i] % b.stripW - b.left]++
        val lo = max(1, (b.width * 0.3).toInt())
        val hi = min(b.width - 2, (b.width * 0.7).toInt())
        if (hi < lo) return listOf(b)
        val mid = b.width / 2
        var cut = lo
        for (c in lo..hi) {
            if (cols[c] < cols[cut] || (cols[c] == cols[cut] && abs(c - mid) < abs(cut - mid))) cut = c
        }
        val l = Blob(b.stripW)
        val r = Blob(b.stripW)
        for (i in 0 until b.px.size) {
            val p = b.px.a[i]
            if (p % b.stripW - b.left <= cut) l.add(p) else r.add(p)
        }
        return if (l.size == 0 || r.size == 0) listOf(b) else listOf(l, r)
    }

    // ------------------------------------------------------------------ 6) crops
    private fun toCrop(b: Blob): FloatArray {
        val bw = b.width
        val bh = b.height
        val src = FloatArray(bw * bh)
        for (i in 0 until b.px.size) {
            val p = b.px.a[i]
            src[(p / b.stripW - b.top) * bw + (p % b.stripW - b.left)] = 1f
        }
        val scale = (SIZE - 2 * PAD).toFloat() / max(bw, bh)
        val nw = max(1, (bw * scale).roundToInt())
        val nh = max(1, (bh * scale).roundToInt())
        val small = resizeArea(src, bw, bh, nw, nh)
        val out = FloatArray(SIZE * SIZE)
        val ox = (SIZE - nw) / 2
        val oy = (SIZE - nh) / 2
        for (y in 0 until nh) for (x in 0 until nw) out[(oy + y) * SIZE + ox + x] = small[y * nw + x]
        return out
    }

    /** Area-weighted resize (every output pixel averages the source pixels it covers). */
    private fun resizeArea(src: FloatArray, sw: Int, sh: Int, dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        val rx = sw.toFloat() / dw
        val ry = sh.toFloat() / dh
        for (ty in 0 until dh) {
            val y0 = ty * ry
            val y1 = (ty + 1) * ry
            for (tx in 0 until dw) {
                val x0 = tx * rx
                val x1 = (tx + 1) * rx
                var sum = 0f
                var weight = 0f
                for (iy in floor(y0).toInt() until min(sh, ceil(y1).toInt())) {
                    val wy = min(iy + 1f, y1) - max(iy.toFloat(), y0)
                    for (ix in floor(x0).toInt() until min(sw, ceil(x1).toInt())) {
                        val wx = min(ix + 1f, x1) - max(ix.toFloat(), x0)
                        sum += src[iy * sw + ix] * wx * wy
                        weight += wx * wy
                    }
                }
                out[ty * dw + tx] = if (weight > 0f) sum / weight else 0f
            }
        }
        return out
    }

    private fun median(v: List<Double>): Double {
        val s = v.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }
}
