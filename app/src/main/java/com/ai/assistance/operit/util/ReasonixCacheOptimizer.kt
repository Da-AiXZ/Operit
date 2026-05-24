package com.ai.assistance.operit.util

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ReasonixCacheOptimizer {
    private val LOG_FILE = File("/storage/emulated/0/1/app制作工具/1/检测", "reasonix_verify.txt")
    
    private fun log(msg: String) {
        try {
            LOG_FILE.parentFile?.mkdirs()
            val ts = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()).format(Date())
            LOG_FILE.appendText("[$ts] $msg\n")
        } catch (_: Exception) {}
    }
    
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
            log("stampMissingReasoning: 跳过 (非thinking模型: $model)")
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
        log("stampMissingReasoning: 补了 $stamped 条消息的 reasoning_content")
        return Pair(out, stamped)
    }
    
    fun dropThinkingMessages(messages: JSONArray): JSONArray {
        var lastUserIdx = -1
        for (i in messages.length() - 1 downTo 0) {
            val role = messages.getJSONObject(i).optString("role")
            if (role == "user" || role == "developer") { lastUserIdx = i; break }
        }
        if (lastUserIdx < 0) {
            log("dropThinkingMessages: 无用户消息, 跳过")
            return messages
        }
        val out = JSONArray()
        var stripped = 0
        var estSavedTokens = 0
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (i < lastUserIdx && msg.optString("role") == "developer") continue
            if (i < lastUserIdx && msg.optString("role") == "assistant") {
                val rc = msg.optString("reasoning_content", "")
                estSavedTokens += rc.length / 4
                val cleaned = JSONObject(msg.toString())
                cleaned.put("reasoning_content", JSONObject.NULL)
                out.put(cleaned)
                stripped++
            } else {
                out.put(msg)
            }
        }
        log("dropThinkingMessages: 剥掉 $stripped 条历史thinking, 省约 ${estSavedTokens} token")
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
        if (healed > 0) log("shrinkOversizedToolResults: 截断 $healed 个超大工具结果")
        else log("shrinkOversizedToolResults: 无超大结果, 跳过")
        return Pair(out, healed)
    }
    
    private fun truncateForModel(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val headLen = (maxChars * 0.80).toInt()
        val tailLen = (maxChars * 0.20).toInt()
        val head = text.take(headLen)
        val tail = text.takeLast(tailLen)
        return head + "\n...[截断: " + text.length + " -> " + maxChars + " 字符, " + text.lines().size + " 行]\n" + tail
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
        if (dropped > 0) log("fixToolCallPairing: 移除 $dropped 条孤立/未配对消息")
        else log("fixToolCallPairing: 无配对问题, 跳过")
        return Pair(out, dropped)
    }
    
    fun healMessagesBeforeSend(messages: JSONArray, model: String, enableDropThinking: Boolean = true): JSONArray {
        val t0 = System.currentTimeMillis()
        val origCount = messages.length()
        log("========== healMessagesBeforeSend 开始 ==========")
        log("  消息数: $origCount  模型: $model  dropThinking: $enableDropThinking")
        
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
        log("  完成: 消息 ${origCount}→${working.length()} | stamp=$stampCount shrink=$shrinkCount drop=$dropCount | 耗时 ${elapsed}ms")
        log("================================================")
        return working
    }
}
