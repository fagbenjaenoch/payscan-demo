package com.example.payscandemo

/*
 * Fully offline NUBAN filter for ML Kit OCR output.
 *
 * Sanity test: bank 058 + account 0000000018 must validate
 *   (sum = 5*7 + 8*3 + 1*3 = 62 -> 10 - 2 = 8).
 */

enum class Confidence { HIGH, MEDIUM, LOW }

data class Bank(val code: String, val name: String, val aliases: List<String>)

data class NubanCandidate(
    val number: String,
    val bank: Bank?,
    val confidence: Confidence,
    val suggestedFix: String? = null,
)

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

    private val patterns: Map<Bank, List<Regex>> = banks.associateWith { bank ->
        bank.aliases.map { Regex("\\b${Regex.escape(it)}\\b") }
    }

    /** Returns a bank only if exactly one bank is mentioned; ambiguous or none -> null. */
    fun findIn(text: String): Bank? {
        val t = text.lowercase()
        return patterns.filter { (_, regexes) -> regexes.any { it.containsMatchIn(t) } }
            .keys.singleOrNull()
    }
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

object NubanExtractor {
    private val CONFUSABLE = mapOf(
        'O' to '0', 'o' to '0', 'I' to '1', 'l' to '1', '|' to '1',
        'S' to '5', 's' to '5', 'B' to '8', 'Z' to '2', 'z' to '2',
    )

    private const val C = "[0-9OoIl|SsBZz]"

    // 10 digit-like characters (optional space/dash/dot between), not touching other letters/digits.
    // The boundary checks also reject 11-digit phone numbers and BVNs.
    private val CANDIDATE =
        Regex("(?<![0-9A-Za-z])$C(?:[ \\-.]?$C){9}(?![0-9A-Za-z])")

    private val KEYWORD = Regex("\\b(account|acct|a/c|nuban)\\b", RegexOption.IGNORE_CASE)

    internal fun normalize(raw: String): String = buildString(raw.length) {
        for (ch in raw) {
            if (ch !in " -.") append(CONFUSABLE[ch] ?: ch)
        }
    }

    fun extract(frameText: String): List<NubanCandidate> {
        val bank = BankDirectory.findIn(frameText)
        val hasKeyword = KEYWORD.containsMatchIn(frameText)
        val out = LinkedHashMap<String, NubanCandidate>()

        for (m in CANDIDATE.findAll(frameText)) {
            val raw = m.value
            // At most 2 characters may be letter lookalikes; otherwise it's probably a word.
            if (raw.count { it.isDigit() } < 8) continue

            val number = normalize(raw)
            if (number.length != 10 || !number.all(Char::isDigit)) continue
            if (number.toSet().size <= 2) continue // 0000000000, 1212121212, ...

            classify(number, bank, hasKeyword)?.let { out.putIfAbsent(number, it) }
        }
        return out.values.toList()
    }

    private fun classify(number: String, bank: Bank?, hasKeyword: Boolean): NubanCandidate? {
        if (bank != null) {
            if (NubanValidator.isValid(bank.code, number)) {
                return NubanCandidate(number, bank, Confidence.HIGH)
            }
            val fix = NubanValidator.singleDigitFix(bank.code, number)
            return if (fix != null || hasKeyword) {
                NubanCandidate(number, bank, Confidence.LOW, suggestedFix = fix)
            } else null
        }
        val passesSomeBank = NubanValidator.banksPassing(number).isNotEmpty()
        return when {
            passesSomeBank && hasKeyword -> NubanCandidate(number, null, Confidence.MEDIUM)
            passesSomeBank || hasKeyword -> NubanCandidate(number, null, Confidence.LOW)
            else -> null
        }
    }
}

/** Accepts a number only after it appears in `required` of the last `window` frames. */
class FrameVoter(private val window: Int = 8, private val required: Int = 3) {
    private val history = ArrayDeque<Map<String, NubanCandidate>>()

    fun submit(found: List<NubanCandidate>): NubanCandidate? {
        history.addLast(found.associateBy { it.number })
        if (history.size > window) history.removeFirst()

        val votes = history.flatMap { it.keys }.groupingBy { it }.eachCount()
        val best = votes.filter { it.value >= required }
            .maxByOrNull { it.value }?.key ?: return null
        return history.lastOrNull { best in it }?.get(best)
    }

    fun reset() {
        history.clear()
    }
}

/**
 * Stateful entry point imported by the UI / CameraX analyzer.
 * Combines single-frame NUBAN extraction, multi-frame consensus voting, and OCR block matching.
 */
class NubanFilter(window: Int = 8, required: Int = 3) {
    private val voter = FrameVoter(window = window, required = required)

    /** Processes full-frame OCR text and returns a candidate once it passes multi-frame voting. */
    fun filter(frameText: String): NubanCandidate? {
        val candidates = if (frameText.isBlank()) {
            emptyList()
        } else {
            NubanExtractor.extract(frameText)
        }
        return voter.submit(candidates)
    }

    /** Returns true if [blockText] contains the confirmed [candidate] account number. */
    fun matchesBlock(blockText: String, candidate: NubanCandidate): Boolean {
        return candidate.number in NubanExtractor.normalize(blockText)
    }

    /** Clears accumulated frame votes when resetting or starting a new scan. */
    fun reset() = voter.reset()
}