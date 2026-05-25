package com.ai.assistance.operit.util

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Reasonix 全量移植 — 14 项 DeepSeek API 消息优化。
 *
 * 移植自 https://github.com/nicobao/deepseek-reasonix (MIT)
 *
 * 流水线顺序（与 Reasonix src/loop.ts 对齐）：
 *   1. stampMissingReasoning       — 补齐缺失的 reasoning_content 防 400
 *   2. dropThinkingMessages        — 剥离历史 thinking 省 token
 *   3. stripHallucinatedToolMarkup — 清理模型输出的 DSML/function_calls 幻觉标记
 *   4. scavengeToolCalls           — 从 reasoning_content/content 中恢复遗漏的 tool_calls
 *   5. repairTruncatedJson         — 修复 API 截断的 tool-call 参数 JSON
 *   6. shrinkOversizedToolCallArgs — 截断 assistant tool_calls 中的超长字符串参数
 *   7. shrinkOversizedToolResults  — 截断超大工具结果 (classic)
 *   8. stormBreak                  — 抑制滑动窗口内重复的 (name,args) 工具调用
 *   9. fixToolCallPairing          — 移除未配对的 tool_calls / 孤立 tool 消息
 *  10. mechanicalTruncate          — 紧急机械截断最旧消息（上下文 >95% 时）
 *  11. trimTrailingToolCalls       — 删除末尾未完成的 assistant+tool_calls
 *
 * 折叠（fold / compactHistory）需要异步 API 调用方能生成 LLM 摘要，
 * 当前版本在同步流水线中无法实现；由 mechanicalTruncate 作为降级替代。
 */
object ReasonixCacheOptimizer {
    private val LOG_FILE = File("/storage/emulated/0/1/app制作工具/1/检测", "reasonix_verify.txt")

    // ── 阈值常量（与 Reasonix 对齐）───────────────────────────────────────────
    private const val DEFAULT_MAX_RESULT_CHARS = 24000
    private const val DEFAULT_MAX_RESULT_TOKENS = 8000
    private const val STORM_WINDOW_SIZE = 6
    private const val STORM_THRESHOLD = 3
    private const val PREFLIGHT_EMERGENCY_THRESHOLD = 0.95
    private const val PREFLIGHT_MECHANICAL_TARGET = 0.70
    const val DEEPSEEK_CTX_TOKENS = 1_000_000  // V4 上下文（对标 Reasonix stats.ts）

    // ── Storm 状态（全局滑动窗口，跨请求持久）────────────────────────────────
    private data class StormEntry(val name: String, val args: String, val readOnly: Boolean)
    private val stormWindow = mutableListOf<StormEntry>()

    @Synchronized
    fun resetStorm() { stormWindow.clear() }

    // ── 日志 ─────────────────────────────────────────────────────────────────
    private fun log(msg: String) {
        try {
            LOG_FILE.parentFile?.mkdirs()
            val ts = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault()).format(Date())
            LOG_FILE.appendText("[$ts] $msg\n")
        } catch (_: Exception) {}
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 工具函数
    // ═══════════════════════════════════════════════════════════════════════════

    fun isThinkingModeModel(model: String): Boolean =
        ("reasoner" in model || model == "deepseek-v4-flash" || model == "deepseek-v4-pro" ||
         (model.contains("deepseek", ignoreCase = true) && model != "deepseek-chat"))

    fun thinkingModeForModel(model: String): String? = when {
        model == "deepseek-chat" -> "disabled"
        "reasoner" in model -> "enabled"
        model == "deepseek-v4-flash" || model == "deepseek-v4-pro" -> "enabled"
        else -> null
    }

    // ── 精确 token 估算（对标 Reasonix tokenizer.ts）───────────────────────
    // 使用真实 DeepSeek V4 BPE tokenizer（从 assets 加载）
    // 对标 countTokensBounded: ≤2048 精确，超限采样头尾
    private var tokenizerLoaded = false

    @Synchronized
    fun ensureTokenizerLoaded(context: android.content.Context) {
        if (tokenizerLoaded) return
        try {
            DeepSeekTokenizer.ensureLoaded(context)
            tokenizerLoaded = true
            log("ensureTokenizerLoaded: DeepSeek V4 BPE tokenizer loaded")
        } catch (e: Exception) {
            log("ensureTokenizerLoaded: failed to load tokenizer (\${e.message}), using heuristic fallback")
        }
    }

    private fun estimateTokens(text: String): Int {
        if (!tokenizerLoaded) {
            // 回退到启发式（首次使用前）
            var ascii = 0; var cjk = 0
            for (ch in text) { if (ch.code <= 127) ascii++ else cjk++ }
            return maxOf((ascii * 0.25 + cjk * 0.55).toInt(), 1)
        }
        return maxOf(DeepSeekTokenizer.countTokens(text), 1)
    }

    private const val BOUNDED_TOKENIZE_CHARS = 2048

    private fun estimateTokensBounded(text: String): Int {
        if (text.isEmpty()) return 0
        if (!tokenizerLoaded || text.length <= BOUNDED_TOKENIZE_CHARS) return estimateTokens(text)
        val headChars = BOUNDED_TOKENIZE_CHARS / 2
        val tailChars = BOUNDED_TOKENIZE_CHARS / 2
        val sample = DeepSeekTokenizer.countTokens(text.take(headChars)) +
                     DeepSeekTokenizer.countTokens(text.takeLast(tailChars))
        val sampleChars = (headChars + tailChars).coerceAtMost(text.length)
        val ratio = sample.toDouble() / sampleChars
        return maxOf((text.length * ratio).toInt(), 1)
    }

    fun estimateMessageTokens(msg: JSONObject): Int {
        val content = msg.optString("content", "")
        val rc = msg.optString("reasoning_content", "")
        val tc = msg.optJSONArray("tool_calls")
        var t = estimateTokensBounded(content) + estimateTokensBounded(rc)
        if (tc != null) {
            for (i in 0 until tc.length()) {
                val call = tc.optJSONObject(i) ?: continue
                t += estimateTokensBounded(call.optJSONObject("function")?.optString("arguments", "") ?: "")
            }
        }
        return maxOf(t, 1)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. stampMissingReasoning — 补齐 reasoning_content
    // ═══════════════════════════════════════════════════════════════════════════
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
        if (stamped > 0) log("stampMissingReasoning: 补了 $stamped 条消息的 reasoning_content")
        return Pair(out, stamped)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. dropThinkingMessages — 剥离历史 thinking
    // ═══════════════════════════════════════════════════════════════════════════
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
                estSavedTokens += estimateTokens(rc)
                val cleaned = JSONObject(msg.toString())
                cleaned.remove("reasoning_content")
                out.put(cleaned)
                stripped++
            } else {
                out.put(msg)
            }
        }
        log("dropThinkingMessages: 剥掉 $stripped 条历史thinking, 省约 ${estSavedTokens} token")
        return out
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. stripHallucinatedToolMarkup — 清理 DSML 幻觉标记
    // ═══════════════════════════════════════════════════════════════════════════
    fun stripHallucinatedToolMarkup(messages: JSONArray): Pair<JSONArray, Int> {
        val out = JSONArray()
        var cleaned = 0
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant") {
                val content = msg.optString("content", "")
                if (content.isNotEmpty()) {
                    var s = content
                    // DSML 全角/半角两种形态
                    s = s.replace(Regex("<[｜|]DSML[｜|]function_calls>[\\s\\S]*?</?[｜|]DSML[｜|]function_calls>"), "")
                    s = s.replace(Regex("<[|]DSML[|]function_calls>[\\s\\S]*?</?[|]DSML[|]function_calls>"), "")
                    s = s.replace(Regex("<function_calls>[\\s\\S]*?</function_calls>"), "")
                    // 未闭合的半截 DSML 开标签
                    s = s.replace(Regex("<[｜|]DSML[｜|][\\s\\S]*$"), "")
                    if (s != content) {
                        cleaned++
                        val newMsg = JSONObject(msg.toString())
                        newMsg.put("content", s.trim())
                        out.put(newMsg)
                        continue
                    }
                }
            }
            out.put(msg)
        }
        if (cleaned > 0) log("stripHallucinatedToolMarkup: 清理了 $cleaned 条幻觉 DSML 标记")
        return Pair(out, cleaned)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 4. scavengeToolCalls — 从 reasoning_content/content 恢复遗漏的 tool_calls
    // ═══════════════════════════════════════════════════════════════════════════
    fun scavengeToolCalls(
        messages: JSONArray,
        allowedNames: Set<String> = emptySet()
    ): Pair<JSONArray, Int> {
        if (messages.length() == 0) return Pair(messages, 0)
        val lastMsg = messages.getJSONObject(messages.length() - 1)
        if (lastMsg.optString("role") != "assistant") return Pair(messages, 0)
        if (lastMsg.has("tool_calls") && lastMsg.optJSONArray("tool_calls")?.length() ?: 0 > 0) {
            return Pair(messages, 0)  // 已有正常的 tool_calls
        }

        val rc = lastMsg.optString("reasoning_content", "")
        val ct = lastMsg.optString("content", "")
        val combined = listOf(rc, ct).filter { it.isNotEmpty() }.joinToString("\n")
        if (combined.isEmpty()) return Pair(messages, 0)

        val recovered = mutableListOf<JSONObject>()

        // 从 JSON 对象中提取 tool-call 签名（三种格式）
        val jsonPattern = Regex("""\{[^}]*?"name"\s*:\s*"([^"]+)"[^}]*?"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\[[^\]]*\]|"[^"]*")[^}]*\}""")
        for (match in jsonPattern.findAll(combined)) {
            val name = match.groupValues[1]
            if (allowedNames.isNotEmpty() && name !in allowedNames) continue
            val args = match.groupValues[2]
            val call = JSONObject().apply {
                put("type", "function")
                put("id", "scavenged-${System.currentTimeMillis()}-${recovered.size}")
                put("function", JSONObject().apply {
                    put("name", name)
                    put("arguments", args)
                })
            }
            recovered.add(call)
        }

        if (recovered.isEmpty()) return Pair(messages, 0)

        val tcArray = JSONArray()
        for (c in recovered) tcArray.put(c)
        val newMsg = JSONObject(lastMsg.toString())
        newMsg.put("tool_calls", tcArray)

        val out = JSONArray()
        for (i in 0 until messages.length() - 1) out.put(messages.getJSONObject(i))
        out.put(newMsg)
        log("scavengeToolCalls: 从文本中恢复了 ${recovered.size} 个 tool_call")
        return Pair(out, recovered.size)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 5. repairTruncatedJson — 修复 API 截断的 tool-call 参数 JSON
    // ═══════════════════════════════════════════════════════════════════════════
    fun repairTruncatedJson(messages: JSONArray): Pair<JSONArray, Int> {
        var repaired = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?: JSONArray()
                var changed = false
                val newCalls = JSONArray()
                for (ci in 0 until calls.length()) {
                    val call = calls.optJSONObject(ci) ?: continue
                    val fn = call.optJSONObject("function")
                    val args = fn?.optString("arguments", "") ?: ""
                    if (args.isEmpty() || args == "{}") {
                        newCalls.put(call)
                        continue
                    }
                    // 快速检查是否合法 JSON
                    try {
                        org.json.JSONObject(args)
                        newCalls.put(call)
                        continue
                    } catch (_: Exception) {}
                    try {
                        org.json.JSONArray(args)
                        newCalls.put(call)
                        continue
                    } catch (_: Exception) {}

                    // 尝试修复
                    val fixed = tryRepairJson(args)
                    if (fixed != args) {
                        changed = true
                        repaired++
                        val newFn = JSONObject(fn.toString())
                        newFn.put("arguments", fixed)
                        val newCall = JSONObject(call.toString())
                        newCall.put("function", newFn)
                        newCalls.put(newCall)
                    } else {
                        newCalls.put(call)
                    }
                }
                if (changed) {
                    val newMsg = JSONObject(msg.toString())
                    newMsg.put("tool_calls", newCalls)
                    out.put(newMsg)
                } else {
                    out.put(msg)
                }
            } else {
                out.put(msg)
            }
        }
        if (repaired > 0) log("repairTruncatedJson: 修复了 $repaired 个截断的 JSON 参数")
        return Pair(out, repaired)
    }

    private fun tryRepairJson(input: String): String {
        var s = input.trim()
        if (s.isEmpty()) return "{}"
        // 修剪尾部逗号
        s = s.replace(Regex(",\\s*$"), "")
        // 补齐未闭合的字符串
        val quoteCount = s.count { it == '"' }
        if (quoteCount % 2 != 0) s += '"'
        // 补齐未闭合的括号
        val openBraces = s.count { it == '{' }
        val closeBraces = s.count { it == '}' }
        val openBrackets = s.count { it == '[' }
        val closeBrackets = s.count { it == ']' }
        s += "}".repeat(maxOf(0, openBraces - closeBraces))
        s += "]".repeat(maxOf(0, openBrackets - closeBrackets))
        // 填充悬挂的 key（"foo": → "foo": null）
        s = s.replace(Regex("\"\\s*:\\s*$"), "\": null")
        return s
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 6. shrinkOversizedToolCallArgs — 截断 assistant tool_calls 中的超长参数
    // ═══════════════════════════════════════════════════════════════════════════
    fun shrinkOversizedToolCallArgs(messages: JSONArray, maxTokens: Int = DEFAULT_MAX_RESULT_TOKENS): Pair<JSONArray, Int> {
        var healed = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?: JSONArray()
                var changed = false
                val newCalls = JSONArray()
                for (ci in 0 until calls.length()) {
                    val call = calls.optJSONObject(ci) ?: continue
                    val fn = call.optJSONObject("function")
                    val args = fn?.optString("arguments", "") ?: ""
                    if (args.length <= maxTokens) {
                        newCalls.put(call)
                        continue
                    }
                    val shrunk = shrinkJsonLongStrings(args)
                    if (shrunk != args) {
                        changed = true
                        healed++
                        val newFn = JSONObject(fn.toString())
                        newFn.put("arguments", shrunk)
                        val newCall = JSONObject(call.toString())
                        newCall.put("function", newFn)
                        newCalls.put(newCall)
                    } else {
                        newCalls.put(call)
                    }
                }
                if (changed) {
                    val newMsg = JSONObject(msg.toString())
                    newMsg.put("tool_calls", newCalls)
                    out.put(newMsg)
                } else {
                    out.put(msg)
                }
            } else {
                out.put(msg)
            }
        }
        if (healed > 0) log("shrinkOversizedToolCallArgs: 截断了 $healed 个超长 tool_call 参数")
        return Pair(out, healed)
    }

    private fun shrinkJsonLongStrings(jsonStr: String): String {
        val parsed: Any?
        try {
            parsed = org.json.JSONTokener(jsonStr).nextValue()
        } catch (_: Exception) {
            val head = jsonStr.take(200)
            return "$head…[shrunk: ${jsonStr.length} chars, unparsed]"
        }
        if (parsed !is JSONObject) return jsonStr
        val LONG_THRESHOLD = 300
        var changed = false
        val out = JSONObject()
        for (key in parsed.keys()) {
            val v = parsed.opt(key)
            if (v is String && v.length > LONG_THRESHOLD) {
                changed = true
                val newlines = v.count { it == '\n' }
                out.put(key, "[…shrunk: ${v.length} chars, $newlines lines — tool already responded, see result]")
            } else {
                out.put(key, v)
            }
        }
        return if (changed) out.toString() else jsonStr
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 7. shrinkOversizedToolResults — 截断超大工具结果
    // ═══════════════════════════════════════════════════════════════════════════
    fun shrinkOversizedToolResults(messages: JSONArray, maxChars: Int = DEFAULT_MAX_RESULT_CHARS): Pair<JSONArray, Int> {
        var healed = 0
        val out = JSONArray()
        for (i in 0 until messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") != "tool") { out.put(msg); continue }
            val content = msg.optString("content", "")
            if (content.length <= maxChars) { out.put(msg); continue }
            healed++
            val headLen = (maxChars * 0.80).toInt()
            val tailLen = (maxChars * 0.20).toInt()
            val truncated = content.take(headLen) +
                "\n...[截断: ${content.length} -> $maxChars 字符, ${content.lines().size} 行]\n" +
                content.takeLast(tailLen)
            val newMsg = JSONObject(msg.toString())
            newMsg.put("content", truncated)
            out.put(newMsg)
        }
        if (healed > 0) log("shrinkOversizedToolResults: 截断 $healed 个超大工具结果")
        return Pair(out, healed)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 8. stormBreak — 抑制滑动窗口内重复的 (name,args) 工具调用
    // ═══════════════════════════════════════════════════════════════════════════
    @Synchronized
    fun stormBreak(messages: JSONArray): Pair<JSONArray, Int> {
        val lastMsg = if (messages.length() > 0) messages.getJSONObject(messages.length() - 1) else null
        if (lastMsg == null || lastMsg.optString("role") != "assistant" || !lastMsg.has("tool_calls")) {
            return Pair(messages, 0)
        }
        val calls = lastMsg.optJSONArray("tool_calls") ?: return Pair(messages, 0)
        if (calls.length() == 0) return Pair(messages, 0)

        var suppressed = 0
        val keptCalls = JSONArray()
        val suppressedCalls = mutableListOf<JSONObject>()

        for (ci in 0 until calls.length()) {
            val call = calls.optJSONObject(ci) ?: continue
            val name = call.optJSONObject("function")?.optString("name", "") ?: ""
            val args = call.optJSONObject("function")?.optString("arguments", "") ?: ""
            if (name.isEmpty()) { keptCalls.put(call); continue }

            // 计算窗口内相同 (name, args) 的数量
            val count = stormWindow.count { it.name == name && it.args == args }
            if (count >= STORM_THRESHOLD - 1) {
                suppressed++
                suppressedCalls.add(call)
            } else {
                keptCalls.put(call)
                stormWindow.add(StormEntry(name, args, false))
                while (stormWindow.size > STORM_WINDOW_SIZE) stormWindow.removeAt(0)
            }
        }

        if (suppressed == 0) return Pair(messages, suppressed)

        // 如果全部被抑制，注入一条 stub tool 结果让模型知道发生了什么
        val out = JSONArray()
        for (i in 0 until messages.length() - 1) out.put(messages.getJSONObject(i))

        if (keptCalls.length() > 0) {
            val newMsg = JSONObject(lastMsg.toString())
            newMsg.put("tool_calls", keptCalls)
            out.put(newMsg)
        } else {
            // 全部抑制：恢复原始 tool_calls + 注入 stub
            out.put(lastMsg)
            for (call in suppressedCalls) {
                val callId = call.optString("id", "storm-suppressed")
                val callName = call.optJSONObject("function")?.optString("name", "") ?: "unknown"
                out.put(JSONObject().apply {
                    put("role", "tool")
                    put("tool_call_id", callId)
                    put("name", callName)
                    put("content", "[repeat-loop guard] this call was suppressed — identical (name,args) repeated $suppressed times. Try a meaningfully different approach, or stop and answer if you have enough.")
                })
            }
        }
        log("stormBreak: 抑制了 $suppressed 个重复 tool_call")
        return Pair(out, suppressed)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 9. fixToolCallPairing — 移除未配对的 tool_calls / 孤立 tool 消息
    // ═══════════════════════════════════════════════════════════════════════════
    fun fixToolCallPairing(messages: JSONArray): Pair<JSONArray, Int> {
        val out = JSONArray()
        var dropped = 0
        var i = 0
        while (i < messages.length()) {
            val msg = messages.getJSONObject(i)
            if (msg.optString("role") == "assistant" && msg.has("tool_calls")) {
                val calls = msg.optJSONArray("tool_calls") ?: JSONArray()
                if (calls.length() == 0) { out.put(msg); i++; continue }
                // stamp missing ids
                val stampedCalls = JSONArray()
                for (ci in 0 until calls.length()) {
                    val call = calls.optJSONObject(ci) ?: continue
                    if (!call.has("id") || call.optString("id", "").isEmpty()) {
                        call.put("id", "z-ext-${System.currentTimeMillis()}-$ci")
                    }
                    stampedCalls.put(call)
                }
                val needed = mutableSetOf<String>()
                for (ci in 0 until stampedCalls.length()) {
                    val id = stampedCalls.optJSONObject(ci)?.optString("id", "")?.trim() ?: ""
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
                    val fixedMsg = JSONObject(msg.toString())
                    fixedMsg.put("tool_calls", stampedCalls)
                    out.put(fixedMsg)
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
        return Pair(out, dropped)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 10. mechanicalTruncate — 紧急机械截断最旧消息（上下文 >95%）
    // ═══════════════════════════════════════════════════════════════════════════
    fun mechanicalTruncate(messages: JSONArray, ctxMax: Int = DEEPSEEK_CTX_TOKENS): Pair<JSONArray, Boolean> {
        val targetTokens = (ctxMax * PREFLIGHT_MECHANICAL_TARGET).toInt()
        // 估算总 token
        var totalTokens = 0
        val tokenCounts = IntArray(messages.length())
        for (i in 0 until messages.length()) {
            tokenCounts[i] = estimateMessageTokens(messages.getJSONObject(i))
            totalTokens += tokenCounts[i]
        }
        if (totalTokens < ctxMax * PREFLIGHT_EMERGENCY_THRESHOLD) {
            return Pair(messages, false)  // 不需要截断
        }

        // 从尾部累计，找到安全的截断点（以 user 消息为边界）
        var cumTokens = 0
        var boundary = messages.length()
        var foundUserBoundary = false
        for (i in messages.length() - 1 downTo 0) {
            if (cumTokens + tokenCounts[i] > targetTokens) break
            cumTokens += tokenCounts[i]
            if (messages.getJSONObject(i).optString("role") == "user") {
                boundary = i
                foundUserBoundary = true
            }
        }
        if (boundary <= 0 || !foundUserBoundary) return Pair(messages, false)

        val out = JSONArray()
        for (i in boundary until messages.length()) {
            out.put(messages.getJSONObject(i))
        }
        val before = messages.length()
        log("mechanicalTruncate: 紧急截断! 消息 ${before}→${out.length()}, " +
            "预计 token ${totalTokens}→${totalTokens - cumTokens} (目标 ≤${targetTokens})")
        return Pair(out, true)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 11. trimTrailingToolCalls — 删除末尾未完成的 assistant+tool_calls
    // ═══════════════════════════════════════════════════════════════════════════
    fun trimTrailingToolCalls(messages: JSONArray): Pair<JSONArray, Boolean> {
        if (messages.length() == 0) return Pair(messages, false)
        val lastMsg = messages.getJSONObject(messages.length() - 1)
        if (lastMsg.optString("role") != "assistant") return Pair(messages, false)
        val calls = lastMsg.optJSONArray("tool_calls")
        if (calls == null || calls.length() == 0) return Pair(messages, false)
        // 检查后续是否有对应的 tool 结果
        // (在消息末尾且无配对 → 删除)
        val out = JSONArray()
        for (i in 0 until messages.length() - 1) out.put(messages.getJSONObject(i))
        log("trimTrailingToolCalls: 删除了末尾未配对的 assistant+tool_calls")
        return Pair(out, true)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 主流水线 — healMessagesBeforeSend
    // ═══════════════════════════════════════════════════════════════════════════
    fun healMessagesBeforeSend(
        messages: JSONArray,
        model: String,
        enableDropThinking: Boolean = true,
        ctxMax: Int = DEEPSEEK_CTX_TOKENS,
        allowedToolNames: Set<String> = emptySet()
    ): JSONArray {
        val t0 = System.currentTimeMillis()
        val origCount = messages.length()
        log("========== healMessagesBeforeSend 开始 ==========")
        log("  消息数: $origCount  模型: $model  dropThinking: $enableDropThinking  ctxMax: $ctxMax")

        var working = messages
        var stampCount = 0
        var shrinkResultCount = 0
        var shrinkArgsCount = 0
        var dropCount = 0
        var stripCount = 0
        var scavengeCount = 0
        var repairCount = 0
        var stormCount = 0
        var truncateCount = 0
        var trimmedCallCount = 0

        // Step 1: stampMissingReasoning
        run {
            val (out, n) = stampMissingReasoning(working, model)
            working = out; stampCount = n
        }

        // Step 2: dropThinkingMessages
        if (enableDropThinking) {
            working = dropThinkingMessages(working)
        }

        // Step 3: stripHallucinatedToolMarkup
        run {
            val (out, n) = stripHallucinatedToolMarkup(working)
            working = out; stripCount = n
        }
        // (cleaned count reported internally)

        // Step 4: scavengeToolCalls
        run {
            val (out, n) = scavengeToolCalls(working, allowedToolNames)
            working = out; scavengeCount = n
        }

        // Step 5: repairTruncatedJson
        run {
            val (out, n) = repairTruncatedJson(working)
            working = out; repairCount = n
        }

        // Step 6: shrinkOversizedToolCallArgs
        run {
            val (out, n) = shrinkOversizedToolCallArgs(working)
            working = out; shrinkArgsCount = n
        }

        // Step 7: shrinkOversizedToolResults
        run {
            val (out, n) = shrinkOversizedToolResults(working)
            working = out; shrinkResultCount = n
        }

        // Step 8: stormBreak
        run {
            val (out, n) = stormBreak(working)
            working = out; stormCount = n
        }

        // Step 9: fixToolCallPairing
        run {
            val (out, n) = fixToolCallPairing(working)
            working = out; dropCount = n
        }

        // Step 10: mechanicalTruncate (context guard)
        run {
            val (out, truncated) = mechanicalTruncate(working, ctxMax)
            working = out
            if (truncated) truncateCount = 1
        }

        // Step 11: trimTrailingToolCalls
        run {
            val (out, trimmed) = trimTrailingToolCalls(working)
            working = out
            if (trimmed) trimmedCallCount = 1
        }

        val elapsed = System.currentTimeMillis() - t0
        log("  完成: 消息 ${origCount}→${working.length()} | " +
            "stamp=$stampCount strip=$stripCount scavenge=$scavengeCount " +
            "repair=$repairCount shrinkArgs=$shrinkArgsCount shrinkResult=$shrinkResultCount " +
            "storm=$stormCount fix=$dropCount truncate=$truncateCount trim=$trimmedCallCount | " +
            "耗时 ${elapsed}ms")
        log("================================================")
        return working
    }


    // ═══════════════════════════════════════════════════════════════════════════
    

    // ═══════════════════════════════════════════════════════════════════════════
    // cache-first loop — 归一化+哈希缓存命中（对标 Reasonix 同机制）
    // ═══════════════════════════════════════════════════════════════════════════
    private const val CACHE_MAX_SIZE = 16

    private data class CachedResponse(
        val content: String,
        val reasoningContent: String,
        val toolCalls: String  // JSON array string
    )

    // ── File-backed cache (persists across conversations & restarts) ──
    private const val CACHE_FILE = "reasonix_response_cache.json"
    private var cacheDir: File? = null

    /** Thread-safe in-memory LRU; file-backed for persistence. */
    private val responseCache = object : LinkedHashMap<String, CachedResponse>(CACHE_MAX_SIZE, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, CachedResponse>?): Boolean {
            return size > CACHE_MAX_SIZE
        }
    }

    @Synchronized
    private fun ensureCacheLoaded() {
        if (responseCache.isNotEmpty()) return
        val dir = cacheDir ?: return
        val file = File(dir, CACHE_FILE)
        if (!file.exists()) return
        try {
            val arr = JSONArray(file.readText())
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val key = obj.getString("k")
                val cr = CachedResponse(obj.getString("c"), obj.optString("r", ""), obj.optString("t", "[]"))
                responseCache[key] = cr
            }
            log("cacheLoaded: ${responseCache.size} entries from $CACHE_FILE")
        } catch (_: Exception) { file.delete() }
    }

    @Synchronized
    private fun persistCache() {
        val dir = cacheDir ?: return
        val file = File(dir, CACHE_FILE)
        try {
            val arr = JSONArray()
            for ((k, v) in responseCache) {
                arr.put(JSONObject().apply {
                    put("k", k); put("c", v.content)
                    put("r", v.reasoningContent); put("t", v.toolCalls)
                })
            }
            file.writeText(arr.toString())
        } catch (e: Exception) { log("persistCache FAIL: ${e.message}") }
    }

    /** Call once during initialization so cache is ready. */
    fun initCache(context: Context) {
        cacheDir = File(context.filesDir, "reasonix_cache").also { it.mkdirs() }
        ensureCacheLoaded()
    }

    @Synchronized
    fun resetCache() { responseCache.clear() }

    /** 归一化用户消息：去标点、trim、小写 → SHA-256 前16位 */
    private fun normalizeMessageForCache(content: String): String {
        return content.trim().lowercase()
            .replace(Regex("""[\p{Punct}]"""), " ")
            .replace(Regex("""\s+"""), " ")
            .trim()
    }

    private fun hashForCache(normalized: String): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        return digest.digest(normalized.toByteArray()).take(8).joinToString("") { "%02x".format(it) }
    }

    /**
     * 历史去重查找：在 prompt history 中查找与 lastUserContent 匹配的 user 消息，
     * 若其后紧跟 assistant 消息，直接返回缓存的回复（跳过 API 调用）。
     * 跨话题也能命中（只要同一对话历史中有过相同提问）。
     */
    fun historyDedupLookup(lastUserContent: String, history: List<*>): JSONObject? {
        val normalized = normalizeMessageForCache(lastUserContent)
        if (normalized.length < 5) return null
        try {
            for (i in0 until history.size - 1) {
                val turn = history[i] ?: continue
                val content = when (turn) {
                    is com.ai.assistance.operit.core.chat.hooks.PromptTurn -> turn.content
                    else -> try {
                        turn.javaClass.getMethod("getContent").invoke(turn) as? String ?: ""
                    } catch (_: Exception) { continue }
                }
                if (normalizeMessageForCache(content) == normalized) {
                    val next = history.getOrNull(i + 1) ?: continue
                    val nextContent = when (next) {
                        is com.ai.assistance.operit.core.chat.hooks.PromptTurn -> next.content
                        else -> try {
                            next.javaClass.getMethod("getContent").invoke(next) as? String ?: ""
                        } catch (_: Exception) { continue }
                    }
                    if (nextContent.isNotEmpty()) {
                        val msg = org.json.JSONObject().apply {
                            put("role", "assistant")
                            put("content", nextContent)
                        }
                        log("historyDedup: HIT at index $i")
                        return msg
                    }
                }
            }
        } catch (_: Exception) { }
        return null
    }

    /** 查询缓存：命中返回已缓存的 assistant 消息 JSON，未命中返回 null */
    fun cacheLookup(lastUserContent: String): JSONObject? {
        ensureCacheLoaded()
        val normalized = normalizeMessageForCache(lastUserContent)
        if (normalized.length < 5) return null  // 太短不缓存（降低门槛）
        val key = hashForCache(normalized)
        val cached = responseCache[key] ?: return null
        val msg = JSONObject()
        msg.put("role", "assistant")
        msg.put("content", cached.content)
        if (cached.reasoningContent.isNotEmpty()) msg.put("reasoning_content", cached.reasoningContent)
        if (cached.toolCalls.isNotEmpty() && cached.toolCalls != "[]") {
            msg.put("tool_calls", JSONArray(cached.toolCalls))
        }
        log("cacheLookup: HIT for key=$key")
        return msg
    }

    /** 存储缓存：API 响应成功后调用 */
    @Synchronized
    fun cacheStore(lastUserContent: String, assistantMessage: JSONObject) {
        val normalized = normalizeMessageForCache(lastUserContent)
        if (normalized.length < 5) return  // was 20, lowered for short messages
        val key = hashForCache(normalized)
        val content = assistantMessage.optString("content", "")
        val rc = assistantMessage.optString("reasoning_content", "")
        val tc = assistantMessage.optJSONArray("tool_calls")?.toString() ?: "[]"
        responseCache[key] = CachedResponse(content, rc, tc)
        persistCache()
        log("cacheStore: stored key=$key (cache size=${responseCache.size})")
    }

    // fold / compactHistory — LLM 摘要折叠（对标 Reasonix context-manager.ts）
    // ═══════════════════════════════════════════════════════════════════════════
    private const val HISTORY_FOLD_THRESHOLD = 0.50
    private const val HISTORY_FOLD_TAIL_FRACTION = 0.20
    private const val HISTORY_FOLD_AGGRESSIVE_THRESHOLD = 0.70
    private const val HISTORY_FOLD_AGGRESSIVE_TAIL_FRACTION = 0.10
    private const val HISTORY_FOLD_MIN_SAVINGS = 0.30
    const val HISTORY_FOLD_MARKER =
        "[CONVERSATION HISTORY SUMMARY — earlier turns folded for context efficiency]\n\n"

    /** 根据实际 promptTokens 判断是否需要折叠 */
    fun shouldFoldHistory(promptTokens: Int, ctxMax: Int = DEEPSEEK_CTX_TOKENS): Boolean {
        return promptTokens.toDouble() / ctxMax > HISTORY_FOLD_THRESHOLD
    }

    /** 判断是否需要激进折叠 */
    fun shouldAggressiveFold(promptTokens: Int, ctxMax: Int = DEEPSEEK_CTX_TOKENS): Boolean {
        return promptTokens.toDouble() / ctxMax > HISTORY_FOLD_AGGRESSIVE_THRESHOLD
    }

    /** 计算折叠边界：从尾部累积 token，最后一个 user 消息为边界 */
    fun calculateFoldBoundary(messages: JSONArray, tailBudgetTokens: Int): Int {
        val tokenCounts = IntArray(messages.length())
        for (i in 0 until messages.length()) {
            tokenCounts[i] = estimateMessageTokens(messages.getJSONObject(i))
        }
        var cumTokens = 0
        var boundary = messages.length()
        for (i in messages.length() - 1 downTo 0) {
            if (cumTokens + tokenCounts[i] > tailBudgetTokens) break
            cumTokens += tokenCounts[i]
            if (messages.getJSONObject(i).optString("role") == "user") {
                boundary = i
            }
        }
        return boundary
    }

    /** 构建折叠用的 summarization 请求消息 */
    fun buildFoldMessages(headMessages: JSONArray): JSONArray {
        val messages = JSONArray()
        messages.put(JSONObject().apply {
            put("role", "system")
            put("content", "You compress conversation history for a coding agent. Output one prose recap that preserves: the user's overall goal, decisions and conclusions reached, files inspected or modified, important tool results still relevant to ongoing work, and any open todos. Skip turn-by-turn play-by-play. No tool calls, no markdown headings, no SEARCH/REPLACE blocks — plain prose only.")
        })
        for (i in 0 until headMessages.length()) {
            messages.put(headMessages.getJSONObject(i))
        }
        messages.put(JSONObject().apply {
            put("role", "user")
            put("content", "Summarize the conversation above as plain prose. This summary replaces the original turns to free context — make it self-contained.")
        })
        return messages
    }

    /** 将摘要内容包装为 assistant 消息 */
    fun buildFoldResultMessage(summaryContent: String, model: String): JSONObject {
        val content = HISTORY_FOLD_MARKER + summaryContent
        val msg = JSONObject()
        msg.put("role", "assistant")
        msg.put("content", content)
        if (isThinkingModeModel(model)) {
            msg.put("reasoning_content", "")
        }
        return msg
    }

    /** 执行折叠：head 用摘要替换，tail 保留 */
    fun applyFold(
        messages: JSONArray,
        boundary: Int,
        summaryMsg: JSONObject
    ): Pair<JSONArray, FoldResult> {
        val out = JSONArray()
        out.put(summaryMsg)
        for (i in boundary until messages.length()) {
            out.put(messages.getJSONObject(i))
        }
        val result = FoldResult(
            folded = true,
            beforeMessages = messages.length(),
            afterMessages = out.length(),
            summaryChars = summaryMsg.optString("content", "").length
        )
        log("foldHistory: 折叠完成! 消息 ${result.beforeMessages}→${result.afterMessages}, " +
            "摘要字符数 ${result.summaryChars}")
        return Pair(out, result)
    }

    data class FoldResult(
        val folded: Boolean,
        val beforeMessages: Int,
        val afterMessages: Int,
        val summaryChars: Int
    )

}