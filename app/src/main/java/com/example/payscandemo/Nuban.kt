package com.example.payscandemo

/*
 * Fully offline account-number filter for ML Kit OCR output. No Android or ML Kit
 * dependencies, so it can be unit-tested on the JVM.
 *
 * Finds two kinds of numbers:
 *   - BANK_NUBAN: 10-digit commercial bank numbers, verified with the NUBAN checksum
 *   - WALLET:     fintech numbers = Nigerian mobile number minus the leading 0 (no checksum exists)
 *
 * Usage (BUNDLED ML Kit model `com.google.mlkit:text-recognition`, so nothing is downloaded):
 *
 *   val voter = FrameVoter()                       // one per camera session, main thread only
 *
 *   // per frame: one string per recognized line
 *   val matches = NubanExtractor.extractFromLines(lineTexts)   // List<LineMatch>
 *   val stable  = voter.submit(matches.map { it.candidate })   // accepted after 2-3 frames
 *
 * Sanity checks:
 *   "GTBank 0000000018"                    -> BANK_NUBAN, HIGH   (058 + 000000001 -> check digit 8)
 *   "OPay 8031234567"                      -> WALLET, MEDIUM
 *   "Account no: 8031234567"               -> WALLET, MEDIUM     (bare form + keyword)
 *   "Account no: +234 803 123 4567"        -> WALLET, LOW        (phone form + keyword)
 *   "Call 08031234567"                     -> dropped            (just a phone number)
 */

enum class Confidence { HIGH, MEDIUM, LOW }

enum class Kind { BANK_NUBAN, WALLET }

data class Bank(val code: String, val name: String, val aliases: List<String>)

data class Fintech(val name: String, val aliases: List<String>)

data class NubanCandidate(
    val number: String,
    val kind: Kind,
    val confidence: Confidence,
    val bank: Bank? = null,
    val fintech: Fintech? = null,
    val suggestedFix: String? = null,
    /** Handwriting only: the model's confidence (0..1) for each digit of [number], so the UI can highlight weak ones. */
    val digitProbs: List<Float>? = null,
)

/** A candidate plus the index of the OCR line it was found on (so the UI can draw a box around it). */
data class LineMatch(val candidate: NubanCandidate, val lineIndex: Int)

/** Matches whole-word aliases; returns an item only if exactly one item is mentioned. */
internal class AliasIndex<T : Any>(items: List<T>, aliases: (T) -> List<String>) {
    private val patterns: Map<T, List<Regex>> = items.associateWith { item ->
        aliases(item).map { Regex("\\b${Regex.escape(it)}\\b") }
    }

    fun findIn(text: String): T? {
        val t = text.lowercase()
        return patterns.filter { (_, regexes) -> regexes.any { it.containsMatchIn(t) } }
            .keys.singleOrNull()
    }
}

/** Starter list of 3-digit CBN codes. Verify against the current CBN/NIBSS list and keep it updatable. */
object BankDirectory {
    val banks = listOf(
        Bank("044", "Access Bank", listOf("access bank")),
        Bank("023", "Citibank", listOf("citibank")),
        Bank("050", "Ecobank", listOf("ecobank")),
        Bank("070", "Fidelity Bank", listOf("fidelity")),
        Bank("011", "First Bank", listOf("first bank", "firstbank")),
        Bank("214", "FCMB", listOf("fcmb", "first city monument")),
        Bank("058", "GTBank", listOf("gtb", "gtbank", "guaranty trust")),
        Bank("030", "Heritage Bank", listOf("heritage bank")),
        Bank("082", "Keystone Bank", listOf("keystone")),
        Bank("076", "Polaris Bank", listOf("polaris")),
        Bank("101", "Providus Bank", listOf("providus")),
        Bank("221", "Stanbic IBTC", listOf("stanbic")),
        Bank("068", "Standard Chartered", listOf("standard chartered")),
        Bank("232", "Sterling Bank", listOf("sterling")),
        Bank("032", "Union Bank", listOf("union bank")),
        Bank("033", "UBA", listOf("uba", "united bank for africa")),
        Bank("215", "Unity Bank", listOf("unity bank")),
        Bank("035", "Wema Bank", listOf("wema")),
        Bank("057", "Zenith Bank", listOf("zenith")),
        Bank("301", "Jaiz Bank", listOf("jaiz")),
        Bank("102", "Titan Trust Bank", listOf("titan trust")),
        Bank("103", "Globus Bank", listOf("globus")),
        Bank("100", "SunTrust Bank", listOf("suntrust")),
    )

    private val index = AliasIndex(banks) { it.aliases }
    fun findIn(text: String): Bank? = index.findIn(text)
}

/** Fintechs whose account number is the customer's phone number minus the leading 0. Verify and extend. */
object FintechDirectory {
    val fintechs = listOf(
        Fintech("OPay", listOf("opay")),
        Fintech("PalmPay", listOf("palmpay", "palm pay")),
        Fintech("Moniepoint", listOf("moniepoint", "monie point")),
    )

    private val index = AliasIndex(fintechs) { it.aliases }
    fun findIn(text: String): Fintech? = index.findIn(text)
}

object NubanValidator {
    // Weights applied to the 12 digits of: bankCode(3) + first 9 digits of the account.
    private val SEEDS = intArrayOf(3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3)

    // Digits OCR commonly mixes up, used for single-digit correction.
    private val LOOKALIKES = mapOf(
        '0' to "68", '1' to "7", '2' to "7", '3' to "8", '4' to "9",
        '5' to "6", '6' to "058", '7' to "12", '8' to "0369", '9' to "48",
    )

    fun checkDigit(bankCode: String, serial9: String): Int {
        val digits = bankCode + serial9
        var sum = 0
        for (i in digits.indices) sum += (digits[i] - '0') * SEEDS[i]
        val r = 10 - (sum % 10)
        return if (r == 10) 0 else r
    }

    fun isValid(bankCode: String, account: String): Boolean =
        bankCode.length == 3 &&
            account.length == 10 &&
            account.all(Char::isDigit) &&
            checkDigit(bankCode, account.substring(0, 9)) == account[9] - '0'

    /** Banks for which this number is checksum-valid. Weak evidence on its own. */
    fun banksPassing(account: String): List<Bank> =
        BankDirectory.banks.filter { isValid(it.code, account) }

    /** If exactly one lookalike substitution makes the number valid for this bank, return it. */
    fun singleDigitFix(bankCode: String, account: String): String? {
        val fixes = mutableSetOf<String>()
        for (i in account.indices) {
            for (alt in LOOKALIKES[account[i]].orEmpty()) {
                val variant = account.substring(0, i) + alt + account.substring(i + 1)
                if (isValid(bankCode, variant)) fixes += variant
            }
        }
        return fixes.singleOrNull()
    }
}

object WalletNumber {
    // National mobile numbers (without the leading 0) start 70x, 80x, 81x, 90x, 91x.
    // Prefix families change over time; keep this configurable.
    private val PREFIX = Regex("^(70|80|81|90|91)\\d{8}$")

    /** Shape check only (about 6% of random 10-digit numbers pass). There is no checksum. */
    fun isPhoneShaped(number: String): Boolean = PREFIX.matches(number)
}

object NubanExtractor {
    private val CONFUSABLE = mapOf(
        'O' to '0', 'o' to '0', 'I' to '1', 'l' to '1', '|' to '1',
        'S' to '5', 's' to '5', 'B' to '8', 'Z' to '2', 'z' to '2',
    )

    private const val C = "[0-9OoIl|SsBZz]"

    // A 10-digit number can have at most 2 letter lookalikes, so a line needs >= 8 real digits.
    private const val MIN_DIGITS = 8

    // Bare 10 digit-like characters (optional space/dash/dot between), not touching other letters/digits.
    private val CANDIDATE =
        Regex("(?<![0-9A-Za-z])$C(?:[ \\-.]?$C){9}(?![0-9A-Za-z])")

    // Phone forms: 0803..., +234 803..., 234803... (11 or 13 digits). Wallet numbers only.
    private val PHONE_LONG = Regex(
        "(?<![0-9A-Za-z])(?:\\+?234|[0Oo])[ \\-.]?[789]$C(?:[ \\-.]?$C){8}(?![0-9A-Za-z])"
    )

    private val KEYWORD = Regex("\\b(account|acct|a/c|nuban)\\b", RegexOption.IGNORE_CASE)

    private data class Context(val bank: Bank?, val fintech: Fintech?, val hasKeyword: Boolean)

    /** Convenience for tests and single strings. */
    fun extract(frameText: String): List<NubanCandidate> =
        extractFromLines(listOf(frameText)).map { it.candidate }

    /**
     * Scans one frame. Bank / fintech / keyword context is taken from ALL lines (the bank name is
     * usually on a different line from the number), but numbers are matched line by line so each
     * result carries the line it came from. Frames with fewer than 8 digits cost almost nothing.
     */
    fun extractFromLines(lines: List<String>): List<LineMatch> {
        if (lines.isEmpty() || countDigits(lines) < MIN_DIGITS) return emptyList()

        val joined = lines.joinToString("\n")
        val ctx = Context(
            bank = BankDirectory.findIn(joined),
            fintech = FintechDirectory.findIn(joined),
            hasKeyword = KEYWORD.containsMatchIn(joined),
        )

        val seen = HashSet<String>()
        val out = ArrayList<LineMatch>()
        for (i in lines.indices) {
            val line = lines[i]
            if (countDigits(line) < MIN_DIGITS) continue
            for (candidate in scan(line, ctx)) {
                if (seen.add(candidate.number)) out += LineMatch(candidate, i)
            }
        }
        return out
    }

    private fun countDigits(s: String): Int {
        var n = 0
        for (ch in s) if (ch in '0'..'9') n++
        return n
    }

    private fun countDigits(lines: List<String>): Int = lines.sumOf { countDigits(it) }

    private fun scan(text: String, ctx: Context): Collection<NubanCandidate> {
        val out = LinkedHashMap<String, NubanCandidate>()

        // 1) Phone forms (11 or 13 digits): reduce to the 10-digit national number.
        val longRanges = mutableListOf<IntRange>()
        for (m in PHONE_LONG.findAll(text)) {
            longRanges += m.range
            val national = nationalNumber(m.value) ?: continue
            classify(national, bareForm = false, ctx)?.let { out.putIfAbsent(national, it) }
        }

        // 2) Bare 10-digit numbers (skip anything already covered by a phone-form match).
        for (m in CANDIDATE.findAll(text)) {
            if (longRanges.any { it.first <= m.range.last && m.range.first <= it.last }) continue
            val number = normalize(m.value) ?: continue
            if (number.length != 10) continue
            if (number.toSet().size <= 2) continue // 0000000000, 1212121212, ...
            classify(number, bareForm = true, ctx)?.let { out.putIfAbsent(number, it) }
        }
        return out.values
    }

    /** True if [blockText], after dropping separators and fixing letter lookalikes, contains [number]. */
    fun containsNumber(blockText: String, number: String): Boolean {
        val cleaned = blockText.filter { it !in " -.+" }.map { CONFUSABLE[it] ?: it }.joinToString("")
        return number in cleaned
    }

    /** Strips separators, fixes letter lookalikes, and rejects runs with more than 2 letters. */
    private fun normalize(raw: String): String? {
        val chars = raw.filter { it !in " -.+" }
        if (chars.count { it.isDigit() } < chars.length - 2) return null
        return chars.map { CONFUSABLE[it] ?: it }.joinToString("")
            .takeIf { s -> s.all(Char::isDigit) }
    }

    private fun nationalNumber(raw: String): String? {
        val s = normalize(raw) ?: return null
        return when {
            s.length == 13 && s.startsWith("234") -> s.substring(3)
            s.length == 11 && s.startsWith("0") -> s.substring(1)
            else -> null
        }
    }

    private fun classify(number: String, bareForm: Boolean, ctx: Context): NubanCandidate? {
        val bank = ctx.bank

        // Checksum-verified NUBAN always wins when the bank is known.
        if (bareForm && bank != null && NubanValidator.isValid(bank.code, number)) {
            return NubanCandidate(number, Kind.BANK_NUBAN, Confidence.HIGH, bank = bank)
        }

        val phoneShaped = WalletNumber.isPhoneShaped(number)
        if (phoneShaped && ctx.fintech != null) return wallet(number, bareForm, ctx)

        // A commercial bank is named but the checksum failed: likely a misread.
        if (bareForm && bank != null) {
            val fix = NubanValidator.singleDigitFix(bank.code, number)
            return if (fix != null || ctx.hasKeyword) {
                NubanCandidate(number, Kind.BANK_NUBAN, Confidence.LOW, bank = bank, suggestedFix = fix)
            } else null
        }

        if (phoneShaped) return wallet(number, bareForm, ctx)
        if (!bareForm) return null

        // Unknown bank: the checksum passing for some bank is weak evidence only.
        val passesSomeBank = NubanValidator.banksPassing(number).isNotEmpty()
        return when {
            passesSomeBank && ctx.hasKeyword -> NubanCandidate(number, Kind.BANK_NUBAN, Confidence.MEDIUM)
            passesSomeBank || ctx.hasKeyword -> NubanCandidate(number, Kind.BANK_NUBAN, Confidence.LOW)
            else -> null
        }
    }

    private fun wallet(number: String, bareForm: Boolean, ctx: Context): NubanCandidate? {
        val confidence = when {
            ctx.fintech != null -> Confidence.MEDIUM
            bareForm && ctx.hasKeyword -> Confidence.MEDIUM
            bareForm || ctx.hasKeyword -> Confidence.LOW
            else -> return null // 0803... with no context is just a phone number
        }
        return NubanCandidate(number, Kind.WALLET, confidence, fintech = ctx.fintech)
    }
}

/**
 * Temporal filter over the last [window] frames. A number is reported once it has been seen in
 * [requiredHigh] frames (checksum-verified, HIGH) or [required] frames (everything else), and it
 * drops out again as it leaves the window. Not thread-safe: call from one thread only.
 */
class FrameVoter(
    private val window: Int = 6,
    private val requiredHigh: Int = 2,
    private val required: Int = 3,
) {
    private val history = ArrayDeque<Map<String, NubanCandidate>>()

    /** Submit every frame (including empty ones) so stale numbers age out. Returns stable numbers. */
    fun submit(found: List<NubanCandidate>): List<NubanCandidate> {
        history.addLast(found.associateBy { it.number })
        if (history.size > window) history.removeFirst()

        val counts = HashMap<String, Int>()
        val latest = LinkedHashMap<String, NubanCandidate>()
        for (frame in history) {
            for ((number, candidate) in frame) {
                counts[number] = (counts[number] ?: 0) + 1
                latest[number] = candidate // later frames overwrite earlier ones
            }
        }

        return latest.values
            .filter { c ->
                val need = if (c.confidence == Confidence.HIGH) requiredHigh else required
                (counts[c.number] ?: 0) >= need
            }
            .sortedBy { it.confidence.ordinal } // HIGH first
    }

    fun reset() = history.clear()
}


/**
 * Thread-safe single-result wrapper used by the camera screen: one confirmed number at a time.
 * [filter] runs on the analyzer thread, [reset] may be called from the UI thread.
 */
class NubanFilter(window: Int = 6, requiredHigh: Int = 2, required: Int = 3) {
    private val lock = Any()
    private val voter = FrameVoter(window, requiredHigh, required)

    /** Processes full-frame OCR text; returns the best number once it has been stable across frames. */
    fun filter(frameText: String): NubanCandidate? {
        val found = if (frameText.isBlank()) emptyList()
        else NubanExtractor.extractFromLines(frameText.lines()).map { it.candidate }
        return synchronized(lock) { voter.submit(found).firstOrNull() }   // HIGH confidence sorts first
    }

    /** True if [blockText] contains the confirmed [candidate] number. */
    fun matchesBlock(blockText: String, candidate: NubanCandidate): Boolean =
        NubanExtractor.containsNumber(blockText, candidate.number)

    fun reset() = synchronized(lock) { voter.reset() }
}
