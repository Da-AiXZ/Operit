package com.ai.assistance.operit.util

import org.json.JSONArray
import org.json.JSONObject

/**
 * Reasonix cache-first core - ported from DeepSeek-Reasonix (MIT)
 *
 * DeepSeek API prompt caching is prefix-based. Three invariants:
 * 1. Every assistant message MUST carry reasoning_content (even empty string).
 * 2. Strip reasoning_content from historical assistant messages to preserve cache prefix.
 * 3. Shrink oversized tool results to maintain prefix continuity.
 *
 * Reference: DeepSeek-Reasonix/src/loop/{healing.ts, thinking.ts, messages.ts, shrink.ts}
 */
object ReasonixCacheOptimizer {

    /** True when the model emits reasoning_content and requires it round-tripped. */
    fun isThinkingModeModel(model: String): Boolean {
        if ("reasoner" in model) return true
        if (model == "deepseek-v4-flash" || model == "deepseek-v4-pro") return true
        if (model.contains("deepseek", ignoreCase = true) && model != "deepseek-chat") return true
        return false
    }

    /** Pins extra_body.thinking.type; unknown returns null for server default. */
    fun thinkingModeForModel(model: String): String? {
        if (model == "deepseek-chat") return "disabled"
        if ("reasoner" in model) return "enabled"
        if (model == "deepseek-v4-flash" || model == "deepseek-v4-pro") return "enabled"
        return null
    }

    /**
     * Ensure all assistant messages carry reasoning_content.
     * Without this, DeepSeek API returns 400 on thinking-mode models.
     */
    fun stampMissingReasoning(messages: JSONArray, model: String): Pair<JSONArray, Int> {
        if (!isThinkingModeModel(model)) return Pair(messages, 0)
        var stamped = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && !msg.has("reasoning_content")) {
                msg.put("reasoning_content", "")
                stamped++
            }
            out.put(msg)
        }
        return Pair(out, stamped)
    }

    /**
     * Drop reasoning_content from assistant messages before the last user/developer.
     * Historical reasoning_content wastes tokens and risks cache-prefix invalidation.
     */
    fun dropThinkingMessages(messages: JSONArray): JSONArray {
        var lastUserIdx = -1
        for (i in messages.length() - 1 downTo 0) {
            val role = messages.getJSONObject(i).optString("role")
            if (role == "user" || role == "developer") {
                lastUserIdx = i
                break
            }
        }
        if (lastUserIdx < 0) return messages
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (i < lastUserIdx && msg.optString("role") == "developer") continue
            if (i < lastUserIdx && msg.optString("role") == "assistant") {
                val cleaned = JSONObject(msg.toString())
                cleaned.put("reasoning_content", JSONObject.NULL)
                out.put(cleaned)
            } else {
                out.put(msg)
            }
        }
        return out
    }

    private const val DEFAULT_MAX_RESULT_CHARS = 24000

    /** Shrink oversized tool messages to maintain cache prefix continuity. */
    fun shrinkOversizedToolResults(
        messages: JSONArray,
        maxChars: Int = DEFAULT_MAX_RESULT_CHARS
    ): Pair<JSONArray, Int> {
        var healed = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") != "tool") {
                out.put(msg)
                continue
            }
            val content = msg.optString("content", "")
            if (content.length <= maxChars) {
                out.put(msg)
                continue
            }
            healed++
            val truncated = truncateForModel(content, maxChars)
            val newMsg = JSONObject(msg.toString())
            newMsg.put("content", truncated)
            out.put(newMsg)
        }
        return Pair(out, healed)
    }

    private fun truncateForModel(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val headLen = (maxChars * 0.80).toInt()
        val tailLen = (maxChars * 0.20).toInt()
        val head = text.take(headLen)
        val tail = text.takeLast(tailLen)
        return head + "\n...[truncated: " + text.length + " -> " + maxChars + " chars, " +
            text.lines().size + " lines total]\n" + tail
    }

    /** Fix unpaired tool_calls and stray tool messages. */
    fun fixToolCallPairing(messages: JSONArray): Pair<JSONArray, Int> {
        val out = JSONArray()
        var dropped = 0
        var i = 0
        while (i < messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?: JSONArray()
                if (calls.length() == 0) {
                    out.put(msg)
                    i++
                    continue
                }
                val needed = mutableSetOf<String>()
                for (ci in 0 until calls.length()) {
                    val call = calls.optJSONObject(ci)
                    val id = call?.optString("id", "")?.trim() ?: ""
                    if (id.isNotEmpty()) needed.add(id)
                }
                if (needed.isEmpty()) {
                    out.put(msg)
                    i++
                    continue
                }
                val candidates = mutableListOf<JSONObject>()
                var j = i + 1
                while (j < messages.length() && needed.isNotEmpty()) {
                    val nxt = messages.getJSONObject(j)
                    if (nxt.optString("role") != "tool") break
                    val id = nxt.optString("tool_call_id", "").trim()
                    if (id !in needed) break
                    needed.remove(id)
                    candidates.add(nxt)
                    j++
                }
                if (needed.isEmpty()) {
                    out.put(msg)
                    candidates.forEach { out.put(it) }
                    i = j - 1
                } else {
                    dropped++
                    i = j - 1
                }
            } else if (msg.optString("role") == "tool") {
                dropped++
            } else {
                out.put(msg)
            }
            i++
        }
        return Pair(out, dropped)
    }

    /** One-stop pre-send repair pipeline. */
    fun healMessagesBeforeSend(
        messages: JSONArray,
        model: String,
        enableDropThinking: Boolean = true
    ): JSONArray {
        var working = messages
        val (stamped) = stampMissingReasoning(working, model)
        working = stamped
        if (enableDropThinking) {
            working = dropThinkingMessages(working)
        }
        val (shrunk) = shrinkOversizedToolResults(working)
        working = shrunk
        val (paired) = fixToolCallPairing(working)
        working = paired
        return working
    }
}
