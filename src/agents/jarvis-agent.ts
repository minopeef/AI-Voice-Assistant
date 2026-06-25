import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { textResponseTool } from "../tools/text-response";
import { visionTool } from "../tools/vision-tool";
import { appLauncherTool } from "../tools/app-launcher-tool";
import { cliTool } from "../tools/cli-tool";
import { fileSystemTool } from "../tools/filesystem-tool";
import { fileOrganizerTool } from "../tools/file-organizer-tool";
import { systemInfoTool } from "../tools/system-info-tool";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { getAssistantPrompt } from "../prompts/prompt-manager";
import { Logger } from "../core/logger";
import { appLauncherService } from "../services/app-launcher-service";
import { smartBrowserService } from "../services/smart-browser-service";

type UserContext = { userId?: string; displayName?: string; email?: string };

const CORE_TOOLS = [
  textResponseTool,
  appLauncherTool,
  cliTool,
  visionTool,
];

export class JarvisAgent {
  private openaiAgent: ReturnType<typeof createReactAgent>;
  private checkpointSaver = new MemorySaver();
  private geminiKey: string | null = null;
  private openaiKey: string;

  constructor(openaiKey: string, geminiKey?: string) {
    this.openaiKey = openaiKey;
    this.geminiKey = geminiKey || null;

    appLauncherService.initializeAIParser(openaiKey, geminiKey);
    smartBrowserService.initializeAIParser(openaiKey, geminiKey);

    const llm = new ChatOpenAI({
      apiKey: openaiKey,
      model: "gpt-4o-mini",
      temperature: 0.3,
      maxTokens: 1024,
      timeout: 8000,
      maxRetries: 1
    });

    this.openaiAgent = createReactAgent({
      llm,
      tools: CORE_TOOLS,
      checkpointSaver: this.checkpointSaver,
      messageModifier: async (messages) => {
        const systemPrompt = `${getAssistantPrompt()}

Available Tools (use sparingly):
- textResponseTool: Text generation, conversations, writing
- appLauncherTool: Open/launch applications and websites
- cliTool: Terminal commands and system operations
- visionTool: Screenshot analysis and image processing

SPEED PRIORITY: Prefer textResponseTool for most queries unless specific tool functionality is clearly needed.`;

        return [new SystemMessage(systemPrompt), ...messages];
      }
    });
  }

  async processQuery(
    query: string,
    sessionId: string,
    userContext?: UserContext
  ): Promise<string> {
    const startTime = Date.now();
    const cleanQuery = query.replace(/\s+/g, ' ').trim();

    Logger.info(`🤖 [JarvisAgent] Processing query: "${cleanQuery}"`);
    Logger.info(`🔑 [JarvisAgent] Session ID: ${sessionId}`);

    const isFastLaneTask = this.detectFastLaneTask(cleanQuery);

    if (isFastLaneTask) {
      Logger.info(`⚡ [FastLane] Using optimized path for text generation`);
      return this.processFastLaneQuery(cleanQuery, userContext, startTime);
    }

    Logger.info(`🔧 [FullAgent] Using complete agent with tools`);
    return this.processComplexQuery(cleanQuery, sessionId, userContext, startTime);
  }

  private detectFastLaneTask(query: string): boolean {
    const lowerQuery = query.toLowerCase();

    const toolRequiredKeywords = [
      'open ', 'launch ', 'start ', 'run ', 'execute',
      'screenshot', 'capture screen', 'take picture',
      'find file', 'search file', 'read file', 'write file',
      'terminal', 'command line', 'kill process'
    ];

    if (toolRequiredKeywords.some(keyword => lowerQuery.includes(keyword))) {
      return false;
    }

    return true;
  }

  private async processFastLaneQuery(
    query: string,
    userContext?: UserContext,
    startTime?: number
  ): Promise<string> {
    try {
      const fastLLM = new ChatOpenAI({
        apiKey: this.openaiKey,
        model: "gpt-4o-mini",
        temperature: 0.1,
        maxTokens: 512,
        timeout: 3000,
        maxRetries: 1,
        topP: 0.8,
        frequencyPenalty: 0,
        presencePenalty: 0
      });

      let systemPrompt = `You are Jarvis. Be helpful and concise.`;
      if (userContext?.displayName) {
        systemPrompt += ` User: ${userContext.displayName}.`;
      }

      const response = await fastLLM.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage(query)
      ]);

      const result = response.content as string;
      const processingTime = Date.now() - (startTime || Date.now());
      Logger.info(`⚡ [FastLane] Query processed in ${processingTime}ms`);

      return result || "I processed your request but couldn't generate a response. Please try again.";
    } catch (error) {
      Logger.warning(`⚡ [FastLane] Failed, falling back to full agent:`, error);
      return this.processComplexQuery(query, `fallback_${Date.now()}`, userContext, startTime);
    }
  }

  private async processComplexQuery(
    query: string,
    sessionId: string,
    userContext?: UserContext,
    startTime?: number
  ): Promise<string> {
    try {
      const result = await this.openaiAgent.invoke(
        {
          messages: [
            ...(userContext?.displayName || userContext?.email
              ? [new SystemMessage(
                  `${getAssistantPrompt()}\n\nUser Context:` +
                  (userContext.displayName ? `\n- Name: ${userContext.displayName}` : '') +
                  (userContext.email ? `\n- Email: ${userContext.email}` : '')
                )]
              : []),
            new HumanMessage(query)
          ]
        },
        {
          configurable: {
            thread_id: sessionId
          }
        }
      );

      const messages = result.messages || [];
      const lastMessage = messages[messages.length - 1];

      let response = '';
      if (lastMessage?.content) {
        response = typeof lastMessage.content === 'string'
          ? lastMessage.content
          : lastMessage.content.map((c: any) => c.text || '').join(' ');
      }

      if (!response) {
        response = "I processed your request but couldn't generate a response. Please try again.";
      }

      const processingTime = Date.now() - (startTime || Date.now());
      Logger.info(`🔧 [FullAgent] Query processed successfully in ${processingTime}ms`);

      return response;
    } catch (error) {
      Logger.error('❌ [JarvisAgent] Error processing query:', error);

      if (error instanceof Error && error.message.includes('API key')) {
        return "I need a valid API key to process your request. Please check your OpenAI API key in settings.";
      } else if (error instanceof Error && error.message.includes('timeout')) {
        return "The request took too long to process. Please try again with a simpler query.";
      } else {
        return "I encountered an error while processing your request. Please try again.";
      }
    }
  }

  analyzeApplicationContext(query: string): { targetApp?: string; searchQuery?: string; isSearch: boolean } {
    const lowerQuery = query.toLowerCase();

    const searchIndicators = ['find', 'search', 'look for', 'show me', 'list', 'what'];
    const isSearch = searchIndicators.some(indicator => lowerQuery.includes(indicator));

    const appKeywords: Record<string, string[]> = {
      'browser': ['chrome', 'safari', 'firefox', 'browser', 'web'],
      'music': ['spotify', 'music', 'itunes', 'apple music'],
      'chat': ['slack', 'discord', 'messages', 'whatsapp', 'telegram'],
      'code': ['vscode', 'visual studio', 'code', 'xcode', 'sublime'],
      'terminal': ['terminal', 'iterm', 'console'],
      'notes': ['notes', 'notion', 'obsidian', 'bear'],
      'email': ['mail', 'gmail', 'outlook'],
      'calendar': ['calendar', 'cal', 'fantastical'],
      'video': ['zoom', 'meet', 'teams', 'facetime']
    };

    for (const [, keywords] of Object.entries(appKeywords)) {
      for (const keyword of keywords) {
        if (lowerQuery.includes(keyword)) {
          return {
            targetApp: keyword,
            searchQuery: isSearch ? query : undefined,
            isSearch
          };
        }
      }
    }

    return { searchQuery: query, isSearch: true };
  }

  async clearMemory(sessionId?: string): Promise<void> {
    try {
      if (sessionId) {
        Logger.debug(`🧹 Clearing memory for session: ${sessionId}`);
      } else {
        Logger.debug('🧹 Clearing all agent memory');
        this.checkpointSaver = new MemorySaver();

        const llm = new ChatOpenAI({
          apiKey: this.openaiKey,
          model: "gpt-4o-mini",
          temperature: 0.3,
          maxTokens: 1024,
          timeout: 10000
        });

        this.openaiAgent = createReactAgent({
          llm,
          tools: CORE_TOOLS,
          checkpointSaver: this.checkpointSaver,
        });
      }
      Logger.info('✅ Memory cleared successfully');
    } catch (error) {
      Logger.error('❌ Failed to clear memory:', error);
    }
  }
}
