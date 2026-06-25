/**
 * Unified Agent
 * A clean, provider-agnostic agent that uses native function calling
 */

import { LLMProvider, LLMResponse } from '../core/llm-provider';
import { ToolRegistry, createDefaultToolRegistry } from '../tools/tool-registry';
import { Logger } from '../core/logger';

type UserContext = { userId?: string; displayName?: string; email?: string };

const BASE_SYSTEM_PROMPT = `You are Jarvis, a helpful AI assistant. You have access to tools to help the user.

When the user asks you to do something that requires a tool, use the appropriate tool.
When the user asks a general question or wants text content, respond directly without using tools.

Be concise and direct in your responses. Do not add unnecessary preamble or follow-up questions.`;

function buildSystemPrompt(userContext?: UserContext): string {
  if (!userContext?.displayName && !userContext?.email) return BASE_SYSTEM_PROMPT;
  let context = '\n\nUser Context:';
  if (userContext.displayName) context += `\n- Name: ${userContext.displayName}`;
  if (userContext.email) context += `\n- Email: ${userContext.email}`;
  return BASE_SYSTEM_PROMPT + context;
}

export class UnifiedAgent {
  private provider: LLMProvider;
  private toolRegistry: ToolRegistry;

  constructor(provider: LLMProvider, toolRegistry?: ToolRegistry) {
    this.provider = provider;
    this.toolRegistry = toolRegistry || createDefaultToolRegistry();
    Logger.info(`🤖 [UnifiedAgent] Initialized with ${this.provider.name} provider and ${this.toolRegistry.size} tools`);
  }

  async processQuery(
    query: string,
    sessionId?: string,
    userContext?: UserContext
  ): Promise<string> {
    const startTime = Date.now();
    const preview = query.length > 50 ? `${query.substring(0, 50)}...` : query;

    try {
      Logger.info(`📥 [UnifiedAgent] Processing query: "${preview}"`);

      if (this.provider.supportsToolCalling()) {
        return await this.processWithTools(query, userContext, startTime);
      }

      return await this.processTextOnly(query, userContext, startTime);
    } catch (error) {
      Logger.error('❌ [UnifiedAgent] Query processing failed:', error);
      return 'I encountered an error processing your request. Please try again.';
    }
  }

  private async processWithTools(query: string, userContext: UserContext | undefined, startTime: number): Promise<string> {
    const tools = this.toolRegistry.getToolDefinitions();
    const systemPrompt = buildSystemPrompt(userContext);

    try {
      const response = await this.provider.callWithTools(query, tools, systemPrompt);

      if (response.type === 'tool_call' && response.toolCall) {
        Logger.info(`🔧 [UnifiedAgent] Executing tool: ${response.toolCall.name}`);
        const result = await this.toolRegistry.execute(
          response.toolCall.name,
          response.toolCall.arguments
        );

        const elapsed = Date.now() - startTime;
        Logger.info(`✅ [UnifiedAgent] Tool execution completed in ${elapsed}ms`);
        return result;
      }

      const elapsed = Date.now() - startTime;
      Logger.info(`✅ [UnifiedAgent] Text response in ${elapsed}ms`);
      return response.text || 'I could not generate a response.';

    } catch (error) {
      Logger.error('❌ [UnifiedAgent] Tool calling failed, falling back to text:', error);
      return this.processTextOnly(query, userContext, startTime);
    }
  }

  private async processTextOnly(query: string, userContext: UserContext | undefined, startTime: number): Promise<string> {
    const systemPrompt = buildSystemPrompt(userContext);
    const response = await this.provider.generateText(query, systemPrompt);
    const elapsed = Date.now() - startTime;
    Logger.info(`✅ [UnifiedAgent] Text-only response in ${elapsed}ms`);
    return response;
  }

  async clearMemory(sessionId?: string): Promise<void> {
    Logger.info(`🧹 [UnifiedAgent] Memory cleared`);
  }
}
