package com.example.payscandemo

/** Anything that turns 32x32 crops into 11 probabilities each: digits 0-9, then "not a digit". */
interface DigitModel {
    /** Each crop is a FloatArray(1024), values 0..1, ink = 1. Returns one FloatArray(11) per crop. */
    fun predict(crops: List<FloatArray>): List<FloatArray>
}

/** What the user picked in the UI (bank or fintech chip). Both null = not told. */
data class HandwritingContext(val bank: Bank? = null, val fintech: Fintech? = null)

enum class ReadStatus { OK, NO_DIGITS, WRONG_COUNT, NOT_DIGITS, BAD_PREFIX }

class HandwritingResult(
    val status: ReadStatus,
    val candidate: NubanCandidate?,
    /** Digit boxes inside the strip (strip pixels), for debugging or drawing. */
    val boxes: List<DigitBox> = emptyList(),
    val note: String? = null,
)

/**
 * strip -> digits -> number. Call [read] once per camera frame from ONE thread (the analyzer thread).
 * Probabilities are averaged over the last few frames, which cuts random misreads a lot.
 */
class HandwritingReader(
    private val model: DigitModel,
    private val averageFrames: Int = 4,
    private val expected: IntRange = 10..11,   // 11 = written with the leading 0 (phone style)
) {
    private val history = ArrayDeque<List<FloatArray>>()

    @Synchronized
    fun reset() = history.clear()

    @Synchronized
    fun read(strip: GrayStrip, ctx: HandwritingContext = HandwritingContext()): HandwritingResult {
        val seg = DigitSegmenter.segment(strip, expected)
        if (seg.crops.isEmpty()) {
            history.clear()
            return HandwritingResult(ReadStatus.NO_DIGITS, null, note = seg.note)
        }
        if (seg.crops.size !in expected) {
            history.clear()
            return HandwritingResult(ReadStatus.WRONG_COUNT, null, seg.boxes, seg.note)
        }

        val probs = model.predict(seg.crops)
        if (history.isNotEmpty() && history.last().size != probs.size) history.clear()
        history.addLast(probs)
        while (history.size > averageFrames) history.removeFirst()
        val avg = average(history)

        if (avg.any { it[JUNK] > 0.5f }) {
            return HandwritingResult(ReadStatus.NOT_DIGITS, null, seg.boxes, "a shape does not look like a digit")
        }
        // Renormalise over the 10 digit classes only.
        var digitProbs = avg.map { p ->
            val s = p.take(10).sum().coerceAtLeast(1e-6f)
            FloatArray(10) { p[it] / s }
        }

        if (digitProbs.size == 11) {            // leading 0 written: drop it, keep the 10-digit national form
            if (digitProbs[0].indices.maxByOrNull { digitProbs[0][it] } != 0) {
                return HandwritingResult(ReadStatus.BAD_PREFIX, null, seg.boxes, "11 digits but the first is not 0")
            }
            digitProbs = digitProbs.drop(1)
        }
        return HandwritingResult(ReadStatus.OK, HandwritingInterpreter.interpret(digitProbs, ctx), seg.boxes)
    }

    private fun average(frames: List<List<FloatArray>>): List<FloatArray> {
        val n = frames.first().size
        return List(n) { i -> FloatArray(11) { c -> frames.map { it[i][c] }.average().toFloat() } }
    }

    private companion object {
        const val JUNK = 10
    }
}

/** Turns 10 per-digit probability vectors into a [NubanCandidate] using the NUBAN / phone-number rules. */
object HandwritingInterpreter {
    private const val SURE = 0.6f          // every digit at least this confident -> "consistent" stays HIGH/MEDIUM
    private const val MIN_ALT = 0.10f      // a runner-up digit must have at least this probability to be tried
    private const val WEAK_POSITIONS = 3   // only the least-sure digits are ever changed

    fun interpret(digitProbs: List<FloatArray>, ctx: HandwritingContext): NubanCandidate {
        require(digitProbs.size == 10) { "expected 10 digits" }
        val best = CharArray(10) { i -> ('0'.code + digitProbs[i].indices.maxByOrNull { digitProbs[i][it] }!!).toChar() }
        val number = String(best)
        val bestProbs = List(10) { i -> digitProbs[i].max() }
        val minP = bestProbs.min()

        val bank = ctx.bank
        val wallet = bank == null && (ctx.fintech != null || WalletNumber.isPhoneShaped(number))
        val isValid: (String) -> Boolean = when {
            bank != null -> { s -> NubanValidator.isValid(bank.code, s) }
            wallet -> { s -> WalletNumber.isPhoneShaped(s) }
            else -> { _ -> false }          // unknown bank: nothing to check against
        }

        val kind = if (wallet) Kind.WALLET else Kind.BANK_NUBAN
        if (bank == null && !wallet) {
            return NubanCandidate(number, kind, Confidence.LOW, digitProbs = bestProbs)
        }
        if (isValid(number)) {
            val c = when {
                minP < SURE -> if (wallet) Confidence.LOW else Confidence.MEDIUM
                // No checksum exists for wallets, so never HIGH; MEDIUM only if the user named the fintech.
                wallet -> if (ctx.fintech != null) Confidence.MEDIUM else Confidence.LOW
                else -> Confidence.HIGH
            }
            return NubanCandidate(number, kind, c, bank = bank, fintech = ctx.fintech, digitProbs = bestProbs)
        }
        return NubanCandidate(
            number, kind, Confidence.LOW, bank = bank, fintech = ctx.fintech,
            suggestedFix = suggestFix(best, digitProbs, isValid), digitProbs = bestProbs,
        )
    }

    private class Fix(val number: String, val score: Double)

    /**
     * Tries swapping the 1-2 least-sure digits for their runner-up and keeps swaps that pass [isValid].
     * A number is returned only if one swap clearly beats the others (about 1 random swap in 10 passes
     * a checksum by luck, so a lone weak "winner" is not trusted).
     */
    fun suggestFix(best: CharArray, probs: List<FloatArray>, isValid: (String) -> Boolean): String? {
        val weak = probs.indices.sortedBy { probs[it].max() }.take(WEAK_POSITIONS)
        val alts = weak.associateWith { i ->
            probs[i].indices.filter { it != best[i] - '0' && probs[i][it] >= MIN_ALT }
                .sortedByDescending { probs[i][it] }.take(2)
        }
        fun ratio(i: Int, d: Int) = probs[i][d].toDouble() / probs[i][best[i] - '0'].coerceAtLeast(1e-6f)

        val fixes = ArrayList<Fix>()
        for (a in weak) for (da in alts[a].orEmpty()) {
            val one = best.copyOf().also { it[a] = ('0'.code + da).toChar() }
            if (isValid(String(one))) fixes += Fix(String(one), ratio(a, da))
            for (b in weak) {
                if (b <= a) continue
                for (db in alts[b].orEmpty()) {
                    val two = one.copyOf().also { it[b] = ('0'.code + db).toChar() }
                    if (isValid(String(two))) fixes += Fix(String(two), ratio(a, da) * ratio(b, db))
                }
            }
        }
        fixes.sortByDescending { it.score }
        val top = fixes.firstOrNull() ?: return null
        return if (fixes.size == 1 || top.score >= 3 * fixes[1].score) top.number else null
    }
}
