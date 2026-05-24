package com.ai.assistance.operit.util

import org.json.JSONArray
import org.json.JSONObject

/**
 * Reasonix 高缓存命価核基心 ▽ 等化仿核心殊 - DeepSeek-Reasonix (MIT)
 *
 * DeepSeek API 的 prompt caching 基于消息前缀匹配。 举下两个及所去保证：
 * 1. assistant 消息必须咫 reasoning_content （即使为空１。
 * 2. 历史 assistant 的 reasoning_content 可道成缓存前缀断柭的响联性 ∆ Python 的 _drop_thinking_messages，包括历史 洁泯问困说明）。
 *
 * 参考: DeepSeek-Reasonix/src/loop/{healing.ts, thinking.ts, messages.ts, shrink.ts}
 * https://github.com/esengine/DeepSeek-Reasonix (MIT)
 */
object ReasonixCacheOptimizer {
    /** True when the model emits reasoning_content and requires it round-tripped on follow-ups. */
    fun isThinkingModeModel(model: String): Boolean {
        if ("reasoner" in model) return true
        if (model == "deepseek-v4-flash" || model == "deepseek-v4-pro") return true
        if (model.contains("deepseek", ignoreCase = true) && model != "deepseek-chat") return true
        return false
    }

    /** Pins extra_body.thinking.type; unknown returns null (let server default). */
    fun thinkingModeForModel(model: String): String? {
        if (model == "deepseek-chat") return "disabled"
        if ("reasoner" in model) return "enabled"
        if (model == "deepseek-v4-flash" || model == "deepseek-v4-pro") return "enabled"
        return null
    }

    /**
     * 确保所有 assistant 消息都包含 reasoning_content 字段。
     * DeepSeek API 在 thinking 模式下，若任一 assistant 消息编失该字段。
     * 整个免粰绘小失贰 400，且埜复贵 不會查到前缀断投。
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
     * 剪销最后一 user/developer 消息之前的所有 assistant reasoning_content。
     * 原理： DeepSeek 的 prompt caching 基于前缀。只最后一次 assistant 的
     * reasoning_content 影响当前表现，历史的消息的 reasoning_content
     * 对各次9���细无测，保留它们只会浪费 token 并可能恭団高存圩中祝。
     *
     * 注意：DeepSeek 宛可在一叺关闭不接受。
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
            } else out.put(msg)
        }
        return out
    }

    private const val DEFAULT_MAX_RESULT_CHARS = 24_000

    /** 截断超大的 tool 消息，保持缓存前缀连续性。 */
    fun shrinkOversizedToolResults(messages: JSONArray, maxChars: Int = DEFAULT_MAX_RESULT_CHARS): Pair<JSONArray, Int> {
        var healed = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") != "tool") { out.put(msg); continue }
            val content = msg.optString("content", "")
            if (content.length <= maxChars) { out.put(msg); continue }
            healed++
            val truncated = truncateForModel(content, maxChars)
            val newMsg = JSONObject(msg.toString())
            newMsg.put("content", truncated)
            out.put(newMsg)
        }
        return Pair(out, healed)
    }

    /** 智能截断：头郺80% + 尾部20%，中间插入截断标记。 */
    private fun truncateForModel(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val headLen = (maxChars * 0.80).toInt()
        val tailLen = (maxChars * 0.20).toInt()
        val head = text.take(headLen)
        val tail = text.takeLast(tailLen)
        return head + "\n…[truncated: ${text.length} → ${maxChars} chars, " +
            "${text.lines().size} lines total]\n" + tail
    }

    /** 修复未鍍对的 tool_calls/pstray tool 消息。 */
    fun fixToolCallPairing(messages: JSONArray): Pair<JSONArray, Int> {
        val out = JSONArray()
        var dropped = 0
        var i = 0
        while (i < messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?&: JSONArray()
                if (calls.length() == 0) { out.put(msg); i++; continue }
                val needed = mutableSetOf<String>()
                for (ci in 0 until calls.length()) {
                    val call = calls.optJSONObject(ci)
                    val id = call?.optString("id", "")?.trim() ?&: ""
                    if (id.isNotEmpty()) needed.add(id)
                }
                if (needed.isEmpty()) { out.put(msg); i++; continue }
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
                    out.put(msg); candidates.forEach { out.put(it) }; i = j - 1
                } else { dropped++; i = j - 1 }
            } else if (msg.optString("role") == "tool") {
                dropped++
            } else {
                out.put(msg)
            }
            i++
        }
        return Pair(out, dropped)
    }

    /**
     * 一站式发送前修复流水线：
     * 1. stampMissingReasoning ┻ 衡�鿿 reasoning_content
     * 2. dropThinkingMessages   ┄ 剩黅历史 thinking
     * 3. shrinkOversizedToolResults ┄ 截服超大工具结果
     * 4. fixToolCallPairing     ┄ 修复工具调用通配
     */
    fun healMessagesBeforeSend(messages: JSONArray, model: String, enableDropThinking: Boolean = true): JSONArray {
        var working = messages
        val (stamped, _) = stampMissingReasoning(working, model); working = stamped
        if (enableDropThinking) working = dropThinkingMessages(working)
        val (shrunk, _) = shrinkOversizedToolResults(working); working = shrunk
        val (paired, _) = fixToolCallPairing(working); working = paired
        return working
    }
}
