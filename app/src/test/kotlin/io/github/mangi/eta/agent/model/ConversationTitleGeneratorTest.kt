package io.github.mangi.eta.agent.model

import io.github.mangi.eta.agent.runtime.AgentRunController
import io.github.mangi.eta.data.model.ModelReasoningCapabilities
import io.github.mangi.eta.data.model.ReasoningEffort
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ConversationTitleGeneratorTest {
    private val requests = mutableListOf<ProviderRequest>()

    private fun config(capabilities: ModelReasoningCapabilities? = null) = AgentModelClient.ModelConfig(
        baseUrl = "https://fixture.invalid/v1",
        apiKey = "fixture",
        model = "fixture-small",
        systemPrompt = "",
        reasoningEffort = ReasoningEffort.OFF,
        thinkingEnabled = false,
        reasoningCapabilities = capabilities,
    )

    private fun generator(
        config: AgentModelClient.ModelConfig? = config(),
        timeoutMs: Long = ConversationTitleGenerator.TIMEOUT_MS,
        reply: (AgentRunController, (ProviderEvent) -> Unit) -> ProviderResponse,
    ) = ConversationTitleGenerator(
        loadConfig = { config },
        providerFor = {
            object : AgentProviderClient {
                override val id = "fixture"
                override val capabilities = OpenAiChatCompletionsProvider.capabilities
                override fun complete(
                    request: ProviderRequest,
                    runController: AgentRunController,
                    onEvent: (ProviderEvent) -> Unit,
                ): ProviderResponse = reply(runController, onEvent).also { requests += request }
            }
        },
        timeoutMs = timeoutMs,
    )

    private fun textReply(text: String, finish: String = "stop") = ProviderResponse(
        JSONObject()
            .put("role", "assistant")
            .put("content", text)
            .put("finish_reason", finish),
    )

    @Test
    fun generatesCleanTitleWithoutToolsAndStripsBoilerplate() = runBlocking {
        val promptWithFiles = AgentFileReferencePromptCodec.format(
            request = "帮我分析这个崩溃日志的原因",
            references = listOf(
                AgentFileReference(
                    displayName = "crash.txt",
                    absolutePath = "/sdcard/Download/crash.txt",
                    kind = AgentFileReferenceKind.File,
                ),
            ),
        )
        val title = generator { _, _ -> textReply("「崩溃日志原因分析。」\n补充说明忽略") }
            .generate(
                userPrompt = promptWithFiles,
                assistantReply = "根据堆栈信息，空指针发生在初始化阶段。",
                conversationModelId = "primary-model",
            )

        assertEquals("崩溃日志原因分析", title)
        val request = requests.single()
        assertEquals(ProviderRequestPurpose.TITLE_GENERATION, request.purpose)
        assertEquals(0, request.effectiveTools.length())
        assertEquals(
            ConversationTitleGenerator.SYSTEM_PROMPT,
            request.messages.getJSONObject(0).getString("content"),
        )
        val userPayload = request.messages.getJSONObject(1).getString("content")
        assertTrue(userPayload.contains("帮我分析这个崩溃日志的原因"))
        assertTrue(!userPayload.contains("/sdcard/Download/crash.txt"))
        // 请求正文必须是“待摘要材料”而不是对话历史：否则模型会续写/反驳上一条而不是拟标题。
        assertTrue(userPayload.startsWith("请为下面这段对话拟一个标题。"))
        assertTrue(userPayload.contains("用户："))
        assertTrue(userPayload.contains("助手："))
        assertTrue(!userPayload.contains("conversation_turn"))
        assertTrue(!userPayload.contains("User: "))
        assertTrue(!userPayload.contains("Assistant: "))
        assertTrue(userPayload.endsWith("只输出标题，一行，不要解释。"))
    }

    @Test
    fun sanitizeTitleRemovesPrefixesQuotesMarkdownAndCapsLength() {
        assertEquals("Android Gradle Upgrade", ConversationTitleGenerator.sanitizeTitle("Title: \"Android Gradle Upgrade.\""))
        assertEquals("Jetpack Compose 主题优化", ConversationTitleGenerator.sanitizeTitle("**标题：《Jetpack Compose 主题优化》**"))
        assertEquals("Python JSON Parsing", ConversationTitleGenerator.sanitizeTitle("# `Python JSON Parsing`!"))
        assertNull(ConversationTitleGenerator.sanitizeTitle("   "))
        assertNull(ConversationTitleGenerator.sanitizeTitle("null"))

        val longTitle = "超长标题".repeat(20)
        val sanitized = ConversationTitleGenerator.sanitizeTitle(longTitle)!!
        assertEquals(ConversationTitleGenerator.MAX_TITLE_CHARS, sanitized.length)

        // 不截断代理对高半区
        val emojiBoundary = "A".repeat(ConversationTitleGenerator.MAX_TITLE_CHARS - 1) + "\uD83D\uDE00"
        val truncatedEmoji = ConversationTitleGenerator.sanitizeTitle(emojiBoundary)!!
        assertEquals(ConversationTitleGenerator.MAX_TITLE_CHARS - 1, truncatedEmoji.length)
    }

    @Test
    fun sanitizeTitleCutsAtClauseBoundaryInsteadOfMidSentence() {
        // 超长标题在最近的子句标点处收断，不留半个括号或半个词。
        val longClause = "B站未登录无法读取账号等级，需要先在客户端完成登录后重试"
        val sanitized = ConversationTitleGenerator.sanitizeTitle(longClause)!!
        assertTrue(sanitized.length <= ConversationTitleGenerator.MAX_TITLE_CHARS)
        assertTrue(sanitized.startsWith("B站未登录无法读取账号等级"))
        assertTrue(!sanitized.endsWith("，"))

        // 没有可用子句边界时退化为按宽度硬切，但仍不超限，也不留悬空连接词。
        val noBoundary = "无标点长标题".repeat(10)
        val hardCut = ConversationTitleGenerator.sanitizeTitle(noBoundary)!!
        assertEquals(ConversationTitleGenerator.MAX_TITLE_CHARS, hardCut.length)

        // 停在英文单词中间时退到上一个空格（该串在 MAX_TITLE_CHARS 处正好切在 ASCII 字母上）。
        val englishCut = ConversationTitleGenerator.sanitizeTitle(
            "Eta 群聊功能实用性与 subagent microservices 方案评估",
        )!!
        assertEquals("Eta 群聊功能实用性与 subagent", englishCut)

        // 收断后不能留下未闭合的括号（否则会出现“回放（鹈鹕 vs 马”这种残缺标题）。
        val bracketCut = ConversationTitleGenerator.sanitizeTitle(
            "微博播放马刺11月22日比赛回放（鹈鹕 vs 马刺 全场回放）",
        )!!
        assertTrue(bracketCut.length <= ConversationTitleGenerator.MAX_TITLE_CHARS)
        assertEquals(bracketCut.count { it == '（' }, bracketCut.count { it == '）' })
        assertTrue(!bracketCut.contains("（"))

        // 各类括号同理。
        listOf(
            "打开《众神之地》第四集并跳过片头继续播放",
            "查看「第 1 个福袋的『3选1』弹窗」处的完整截图",
        ).forEach { input ->
            val cut = ConversationTitleGenerator.sanitizeTitle(input)!!
            assertEquals("括号未闭合：$cut", 0, cut.count { it == '（' } - cut.count { it == '）' })
            assertEquals("书名号未闭合：$cut", 0, cut.count { it == '《' } - cut.count { it == '》' })
        }
    }

    @Test
    fun generatedTitleIsNoLongerThanFallbackTitleWidth() {
        // 与首句回退标题（AgentAppState.MAX_TITLE_CHARS = 24）同宽：更长的标题会被列表单行省略号
        // 从任意位置切掉，由生成侧按子句收断更可控。
        assertEquals(24, ConversationTitleGenerator.MAX_TITLE_CHARS)
    }

    @Test
    fun returnsNullWhenDisabledEmptyOrIncomplete() = runBlocking {
        assertNull(generator(config = null) { _, _ -> error("should not invoke") }.generate("hello", "world"))
        assertNull(generator { _, _ -> error("should not invoke") }.generate("  ", "  "))
        assertNull(generator { _, _ -> textReply("半截标题", finish = "length") }.generate("hello", "world"))
    }

    @Test
    fun supportsImageOnlyUserTurnUsingAssistantReply() = runBlocking {
        val title = generator { _, _ -> textReply("植物品种识别") }
            .generate(userPrompt = "", assistantReply = "这张照片里的植物是龟背竹。")
        assertEquals("植物品种识别", title)
        val userPayload = requests.single().messages.getJSONObject(1).getString("content")
        assertTrue(userPayload.contains("[Sent image attachment]"))
        assertTrue(userPayload.contains("龟背竹"))
    }

    @Test
    fun rejectsToolCallsAndCancelsOnTimeout() {
        val toolCallFailure = runCatching {
            runBlocking {
                generator { _, onEvent ->
                    onEvent(ProviderEvent.BlockStart(AssistantBlockKind.TOOL_CALL, 0, "c1", "terminal"))
                    textReply("unused")
                }.generate("hello", "world")
            }
        }.exceptionOrNull()
        assertTrue(toolCallFailure is IllegalStateException)

        val cancelled = CountDownLatch(1)
        val timeoutFailure = runCatching {
            runBlocking {
                generator(timeoutMs = 50L) { controller, _ ->
                    val released = CountDownLatch(1)
                    controller.register {
                        cancelled.countDown()
                        released.countDown()
                    }
                    released.await(30, TimeUnit.SECONDS)
                    textReply("late")
                }.generate("hello", "world")
            }
        }.exceptionOrNull()
        assertTrue(timeoutFailure is TimeoutCancellationException)
        assertTrue(cancelled.await(1, TimeUnit.SECONDS))
    }
}
