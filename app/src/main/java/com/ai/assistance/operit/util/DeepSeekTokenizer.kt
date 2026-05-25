package com.ai.assistance.operit.util

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream

/**
 * DeepSeek V4 BPE Tokenizer — 对标 Reasonix tokenizer.ts。
 * 从 assets/deepseek-tokenizer.json.gz 加载，提供精确 token 计数。
 */
object DeepSeekTokenizer {
    private var loaded = false
    private val vocab = mutableMapOf<String, Int>()
    private val mergeRank = mutableMapOf<String, Int>()
    private val splitRegexes = mutableListOf<Regex>()
    private val byteToChar = buildByteToChar()
    private var addedPattern: Regex? = null
    private val addedMap = mutableMapOf<String, Int>()

    // ── Byte-Level Encoding (GPT-2 byte→char) ──
    private fun buildByteToChar(): Array<String> {
        val result = Array(256) { "" }
        val bs = mutableListOf<Int>()
        for (b in 33..126) bs.add(b)
        for (b in 161..172) bs.add(b)
        for (b in 174..255) bs.add(b)
        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        for (i in bs.indices) {
            result[bs[i]] = String(Character.toChars(cs[i]))
        }
        return result
    }

    @Synchronized
    fun ensureLoaded(context: Context) {
        if (loaded) return
        try {
            val input = context.assets.open("deepseek-tokenizer.json.gz")
            val gzip = GZIPInputStream(input)
            val reader = BufferedReader(InputStreamReader(gzip, "UTF-8"))
            val sb = StringBuilder()
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                sb.append(line).append('\n')
            }
            reader.close()
            val data = org.json.JSONObject(sb.toString())
            val model = data.getJSONObject("model")

            // Vocab
            val vocabObj = model.getJSONObject("vocab")
            for (key in vocabObj.keys()) {
                vocab[key] = vocabObj.getInt(key)
            }

            // Merges → rank map
            val merges = model.getJSONArray("merges")
            for (i in 0 until merges.length()) {
                mergeRank[merges.getString(i)] = i
            }

            // Pre-tokenizer Split patterns
            val preTok = data.getJSONObject("pre_tokenizer")
            val preTokList = preTok.getJSONArray("pretokenizers")
            for (i in 0 until preTokList.length()) {
                val p = preTokList.getJSONObject(i)
                if (p.optString("type") == "Split") {
                    val reStr = p.getJSONObject("pattern").getString("Regex")
                    // Java regex compatibility adjustments
                    val javaRe = reStr
                        .replace("\\p", "\\p")  // Unicode properties pass through
                    splitRegexes.add(Regex(javaRe))
                }
            }

            // Added tokens
            val addedTokens = data.getJSONArray("added_tokens")
            val addedContents = mutableListOf<String>()
            for (i in 0 until addedTokens.length()) {
                val t = addedTokens.getJSONObject(i)
                if (!t.optBoolean("special", true)) {
                    val content = t.getString("content")
                    addedMap[content] = t.getInt("id")
                    addedContents.add(content)
                }
            }
            addedContents.sortByDescending { it.length }
            if (addedContents.isNotEmpty()) {
                addedPattern = Regex(
                    addedContents.joinToString("|") { Regex.escape(it) }
                )
            }

            loaded = true
        } catch (e: Exception) {
            throw RuntimeException("Failed to load DeepSeek tokenizer: ${e.message}", e)
        }
    }

    // ── BPE Encoding ──
    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val ids = mutableListOf<Int>()

        fun process(segment: String) {
            if (segment.isEmpty()) return
            var chunks = mutableListOf(segment)
            for (re in splitRegexes) {
                val next = mutableListOf<String>()
                for (chunk in chunks) {
                    if (chunk.isEmpty()) continue
                    var last = 0
                    for (match in re.findAll(chunk)) {
                        val idx = match.range.first
                        if (idx > last) next.add(chunk.substring(last, idx))
                        if (match.value.isNotEmpty()) next.add(match.value)
                        last = idx + match.value.length
                    }
                    if (last < chunk.length) next.add(chunk.substring(last))
                }
                chunks = next
            }
            for (chunk in chunks) {
                if (chunk.isEmpty()) continue
                val byteLevel = byteLevelEncode(chunk)
                val pieces = bpeEncode(byteLevel)
                for (p in pieces) {
                    val id = vocab[p]
                    if (id != null) ids.add(id)
                }
            }
        }

        val ap = addedPattern
        if (ap != null) {
            var last = 0
            for (match in ap.findAll(text)) {
                val idx = match.range.first
                if (idx > last) process(text.substring(last, idx))
                val id = addedMap[match.value]
                if (id != null) ids.add(id)
                last = idx + match.value.length
            }
            if (last < text.length) process(text.substring(last))
        } else {
            process(text)
        }
        return ids.toIntArray()
    }

    fun countTokens(text: String): Int = encode(text).size

    // ── BPE internals ──
    private fun byteLevelEncode(s: String): String {
        val bytes = s.toByteArray(Charsets.UTF_8)
        val sb = StringBuilder(bytes.size)
        for (b in bytes) {
            sb.append(byteToChar[b.toInt() and 0xFF])
        }
        return sb.toString()
    }

    private fun bpeEncode(piece: String): List<String> {
        if (piece.length <= 1) return if (piece.isNotEmpty()) listOf(piece) else emptyList()
        var word = piece.toCharArray().map { it.toString() }.toMutableList()
        while (true) {
            var bestIdx = -1
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val pair = word[i] + " " + word[i + 1]
                val rank = mergeRank[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                    if (rank ==0) break
                }
            }
            if (bestIdx <0) break
            val merged = word[bestIdx] + word[bestIdx + 1]
            word = (word.take(bestIdx) + merged + word.drop(bestIdx + 2)).toMutableList()
            if (word.size ==1) break
        }
        return word
    }
}
