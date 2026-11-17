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

    // Pre-compiled case-insensitive patterns avoid per-frame text.lowercase() allocations.
    private val patterns: List<Pair<Bank, List<Regex>>> = banks.map { bank ->
        bank to bank.aliases.map { Regex("\\b${Regex.escape(it)}\\b", RegexOption.IGNORE_CASE) }
    }

    /** Returns a bank only if exactly one bank is mentioned; ambiguous or none -> null. */
    fun findIn(text: String): Bank? {
        var matched: Bank? = null
        for ((bank, regexes) in patterns) {
            if (regexes.any { it.containsMatchIn(text) }) {
                if (matched != null) return null
                matched = bank
            }
        }
        return matched
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

    // Precomputed weighted sum for each known 3-digit bank code.
    private val BANK_PREFIX_SUMS: Map<String, Int> = BankDirectory.banks.associate { bank ->
        bank.code to bankPrefixSum(bank.code)
    }

    private fun bankPrefixSum(bankCode: String): Int =
        (bankCode[0] - '0') * 3 + (bankCode[1] - '0') * 7 + (bankCode[2] - '0') * 3

    private fun serialSum(serial9OrAccount: String): Int {
        var sum = 0
        for (i in 0 until 9) {
            sum += (serial9OrAccount[i] - '0') * SEEDS[i + 3]
        }
        return sum
    }

    private fun checkDigitWithSerialSum(bankCode: String, sSum: Int): Int {
        val prefix = BANK_PREFIX_SUMS[bankCode] ?: bankPrefixSum(bankCode)
        val r = 10 - ((prefix + sSum) % 10)
        return if (r == 10) 0 else r
    }

    fun checkDigit(bankCode: String, serial9: String): Int =
        checkDigitWithSerialSum(bankCode, serialSum(serial9))

    fun isValid(bankCode: String, account: String): Boolean =
        bankCode.length == 3 &&
                account.length == 10 &&
                account.all(Char::isDigit) &&
                checkDigitWithSerialSum(bankCode, serialSum(account)) == account[9] - '0'

    /** Banks for which this number is checksum-valid. Weak evidence on its own. */
    fun banksPassing(account: String): List<Bank> {
        if (account.length != 10 || !account.all(Char::isDigit)) return emptyList()
        val sSum = serialSum(account)
        val targetCheck = account[9] - '0'
        return BankDirectory.banks.filter { bank ->
            checkDigitWithSerialSum(bank.code, sSum) == targetCheck
        }
    }

    /** If exactly one lookalike substitution makes the number valid for this bank, return it. */
    fun singleDigitFix(bankCode: String, account: String): String? {
        if (bankCode.length != 3 || account.length != 10) return null
        val chars = account.toCharArray()
        var foundFix: String? = null
        for (i in chars.indices) {
            val orig = chars[i]
            val alts = LOOKALIKES[orig] ?: continue
            for (alt in alts) {
                chars[i] = alt
                val variant = String(chars)
                if (isValid(bankCode, variant)) {
                    if (foundFix != null && foundFix != variant) return null
                    foundFix = variant
                }
            }
            chars[i] = orig
        }
        return foundFix
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

    private fun hasAtMostTwoDistinctChars(s: String): Boolean {
        var first = '\u0000'
        var second = '\u0000'
        var count = 0
        for (ch in s) {
            if (count == 0) {
                first = ch
                count = 1
            } else if (ch != first) {
                if (count == 1) {
                    second = ch
                    count = 2
                } else if (ch != second) {
                    return false
                }
            }
        }
        return true
    }

    fun extract(frameText: String): List<NubanCandidate> {
        var out: LinkedHashMap<String, NubanCandidate>? = null
        var contextEvaluated = false
        var bank: Bank? = null
        var hasKeyword = false

        for (m in CANDIDATE.findAll(frameText)) {
            val raw = m.value
            // At most 2 characters may be letter lookalikes; otherwise it's probably a word.
            if (raw.count { it.isDigit() } < 8) continue

            val number = normalize(raw)
            if (number.length != 10 || !number.all(Char::isDigit)) continue
            if (hasAtMostTwoDistinctChars(number)) continue // 0000000000, 1212121212, ...

            // Evaluate bank/keyword regexes only after at least one valid 10-digit candidate is found.
            if (!contextEvaluated) {
                bank = BankDirectory.findIn(frameText)
                hasKeyword = KEYWORD.containsMatchIn(frameText)
                contextEvaluated = true
            }

            classify(number, bank, hasKeyword)?.let { candidate ->
                val map = out ?: LinkedHashMap<String, NubanCandidate>().also { out = it }
                map.putIfAbsent(number, candidate)
            }
        }
        return out?.values?.toList() ?: emptyList()
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
    private val history = ArrayDeque<Map<String, NubanCandidate>>(window)
    private val voteCounts = HashMap<String, Int>()

    fun submit(found: List<NubanCandidate>): NubanCandidate? {
        if (history.size == window) {
            history.removeFirst()
        }
        history.addLast(if (found.isEmpty()) emptyMap() else found.associateBy { it.number })

        voteCounts.clear()
        for (frame in history) {
            for (key in frame.keys) {
                voteCounts[key] = (voteCounts[key] ?: 0) + 1
            }
        }

        var bestKey: String? = null
        var bestCount = required - 1
        for ((key, count) in voteCounts) {
            if (count > bestCount) {
                bestCount = count
                bestKey = key
            }
        }
        val winner = bestKey ?: return null
        for (i in history.lastIndex downTo 0) {
            history[i][winner]?.let { return it }
        }
        return null
    }

    fun reset() {
        history.clear()
        voteCounts.clear()
    }
}

/**
 * Stateful entry point imported by the UI / CameraX analyzer.
 * Combines single-frame NUBAN extraction, multi-frame consensus voting, and OCR block matching.
 */
class NubanFilter(window: Int = 8, required: Int = 3) {
    private val lock = Any()
    private val voter = FrameVoter(window = window, required = required)

    /** Processes full-frame OCR text and returns a candidate once it passes multi-frame voting. */
    fun filter(frameText: String): NubanCandidate? {
        val candidates = if (frameText.isBlank()) {
            emptyList()
        } else {
            NubanExtractor.extract(frameText)
        }
        return synchronized(lock) {
            voter.submit(candidates)
        }
    }

    /** Returns true if [blockText] contains the confirmed [candidate] account number. */
    fun matchesBlock(blockText: String, candidate: NubanCandidate): Boolean {
        return candidate.number in NubanExtractor.normalize(blockText)
    }

    /** Clears accumulated frame votes when resetting or starting a new scan. */
    fun reset() {
        synchronized(lock) {
            voter.reset()
        }
    }
}