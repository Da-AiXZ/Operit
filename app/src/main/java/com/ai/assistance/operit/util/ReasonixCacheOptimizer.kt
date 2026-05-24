package com.ai.assistance.operit.util

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

object ReasonixCacheOptimizer {
    private const val TAG = "ReasonixCache"
    
    fun isThinkingModeModel(model: String): Boolean = 
        ("reasoner" in model || model == "deepseek-v4-flash" || model == "deepseek-v4-pro" ||
         (model.contains("deepseek", ignoreCase = true) && model != "deepseek-chat"))
    
    fun thinkingModeForModel(model: String): String? = when {
        model == "deepseek-chat" -> "disabled"
        "reasoner" in model -> "enabled"
        model == "deepseek-v4-flash" || model == "deepseek-v4-pro" -> "enabled"
        else -> null
    }
    
    fun stampMissingReasoning(messages: JSONArray, model: String): Pair<JSONArray, Int> {
        if (!isThinkingModeModel(model)) {
            Log.d(TAG, "stampMissingReasoning: skip (not thinking mode: $model)")
            return Pair(messages, 0)
        }
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
        Log.d(TAG, "stampMissingReasoning: patched $stamped messages")
        return Pair(out, stamped)
    }
    
    fun dropThinkingMessages(messages: JSONArray): JSONArray {
        var lastUserIdx = -1
        for (i in messages.length() - 1 downTo 0) {
            val role = messages.getJSONObject(i).optString("role")
            if (role == "user" || role == "developer") { lastUserIdx = i; break }
        }
        if (lastUserIdx < 0) {
            Log.d(TAG, "dropThinkingMessages: no user/developer found, skip")
            return messages
        }
        val out = JSONArray()
        var stripped = 0
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (i < lastUserIdx && msg.optString("role") == "developer") continue
            if (i < lastUserIdx && msg.optString("role") == "assistant") {
                val cleaned = JSONObject(msg.toString())
                cleaned.put("reasoning_content", JSONObject.NULL)
                out.put(cleaned)
                stripped++
            } else {
                out.put(msg)
            }
        }
        Log.d(TAG, "dropThinkingMessages: stripped $stripped historical thinking blocks")
        return out
    }
    
    private const val DEFAULT_MAX_RESULT_CHARS = 24000
    
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
        if (healed > 0) Log.d(TAG, "shrinkOversizedToolResults: shrunk $healed tool results (max ${maxChars} chars)")
        else Log.d(TAG, "shrinkOversizedToolResults: no oversized results")
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
    
    fun fixToolCallPairing(messages: JSONArray): Pair<JSONArray, Int> {
        val out = JSONArray()
        var dropped = 0
        var i = 0
        while (i < messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?: JSONArray()
                if (calls.length() == 0) { out.put(msg); i++; continue }
                val needed = mutableSetOf<String>()
                for (ci in 0 until calls.length()) {
                    val id = calls.optJSONObject(ci)?.optString("id", "")?.trim() ?: ""
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
        if (dropped > 0) Log.w(TAG, "fixToolCallPairing: dropped $dropped unpaired/stray messages")
        else Log.d(TAG, "fixToolCallPairing: no pairing issues")
        return Pair(out, dropped)
    }
    
    fun healMessagesBeforeSend(messages: JSONArray, model: String, enableDropThinking: Boolean = true): JSONArray {
        val t0 = System.currentTimeMillis()
        val origCount = messages.length()
        Log.d(TAG, "healMessagesBeforeSend: IN messages=$origCount model=$model dropThinking=$enableDropThinking")
        
        var working = messages
        val (stamped, stampCount) = stampMissingReasoning(working, model)
        working = stamped
        
        if (enableDropThinking) {
            working = dropThinkingMessages(working)
        }
        
        val (shrunk, shrinkCount) = shrinkOversizedToolResults(working)
        working = shrunk
        
        val (paired, dropCount) = fixToolCallPairing(working)
        working = paired
        
        val elapsed = System.currentTimeMillis() - t0
        Log.d(TAG, "healMessagesBeforeSend: OUT messages=${working.length()} stamp=$stampCount shrink=$shrinkCount drop=$dropCount (${elapsed}ms)")
        return working
    }
}
