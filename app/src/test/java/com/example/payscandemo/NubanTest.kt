package com.example.payscandemo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NubanTest {

    @Test
    fun nubanValidator_checkDigit_calculatesCorrectlyForGTBank() {
        // GTBank code: "058", serial: "000000001" -> check digit should be 8
        val checkDigit = NubanValidator.checkDigit("058", "000000001")
        assertEquals(8, checkDigit)
    }

    @Test
    fun nubanValidator_isValid_validatesCorrectAccount() {
        // Sanity test from docs: bank 058 + account 0000000018
        assertTrue(NubanValidator.isValid("058", "0000000018"))
    }

    @Test
    fun bankDirectory_findIn_identifiesBankNames() {
        val bank = BankDirectory.findIn("Transfer to GTBank account 0000000018")
        assertNotNull(bank)
        assertEquals("058", bank?.code)
        assertEquals("GTBank", bank?.name)
    }

    @Test
    fun nubanExtractor_extract_findsCandidate() {
        val text = "Send payment to GTBank Acct 0000000018"
        val candidates = NubanExtractor.extract(text)

        assertEquals(1, candidates.size)
        assertEquals("0000000018", candidates[0].number)
        assertEquals(Confidence.HIGH, candidates[0].confidence)
    }

    @Test
    fun nubanFilter_voterConsensus_requiresMultipleFrames() {
        val filter = NubanFilter(window = 3, required = 2)
        val text = "GTBank 0000000018"

        val firstFrame = filter.filter(text)
        assertNull(firstFrame) // Only 1 vote, needs 2

        val secondFrame = filter.filter(text)
        assertNotNull(secondFrame) // 2 votes reached
        assertEquals("0000000018", secondFrame?.number)
    }
}