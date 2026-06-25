import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { Logger } from "../core/logger";
import * as os from "os";

import { textResponseTool } from "../tools/text-response";
import { visionTool } from "../tools/vision-tool";
import { appLauncherTool } from "../tools/app-launcher-tool";
import { cliTool } from "../tools/cli-tool";
import { fileSystemTool } from "../tools/filesystem-tool";
import { fileOrganizerTool } from "../tools/file-organizer-tool";
import { systemInfoTool } from "../tools/system-info-tool";
import { CloudLLMService } from "../core/llm-service";

type UserContext = { userId?: string; displayName?: string; email?: string };

/**
 * Tiered Jarvis Agent for optimal performance
 *
 * Tier 1: Direct API (0.5-1s) - 90% of simple text queries
 * Tier 2: Manual Tool Routing (1-3s) - 8% of single tool operations
 * Tier 3: Full Agent (3-5s) - 2% of complex multi-step workflows
 */
export class TieredJarvisAgent {
  private openaiKey: string | null = null;
  private geminiKey: string | null = null;
  private ollamaSettings: { useOllama: boolean; ollamaUrl: string; ollamaModel: string } | null = null;
  private fullAgent: any;
  private llmService: CloudLLMService;
  private checkpointSaver = new MemorySaver();
  private tierCounts = { tier1: 0, tier2: 0, tier3: 0 };

  constructor(
    openaiKey: string | undefined,
    geminiKey?: string,
    ollamaSettings?: { useOllama: boolean; ollamaUrl: string; ollamaModel: string }
  ) {
    this.openaiKey = openaiKey || null;
    this.geminiKey = geminiKey || null;
    this.ollamaSettings = ollamaSettings || null;

    this.llmService = new CloudLLMService(
      openaiKey || '',
      geminiKey || '',
      '',
      ollamaSettings?.useOllama || false,
      ollamaSettings?.ollamaUrl,
      ollamaSettings?.ollamaModel
    );

    if (this.openaiKey) {
      this.initializeFullAgent();
    }
  }

  private initializeFullAgent(): void {
    const llm = new ChatOpenAI({
      apiKey: this.openaiKey,
      model: "gpt-4o-mini",
      temperature: 0.3,
      maxTokens: 1024,
      timeout: 8000,
      maxRetries: 1
    });

    this.fullAgent = createReactAgent({
      llm,
      tools: [
        textResponseTool,
        appLauncherTool,
        cliTool,
        visionTool,
        fileSystemTool,
        fileOrganizerTool,
        systemInfoTool
      ],
      checkpointSaver: this.checkpointSaver
    });
  }

  async processQuery(
    query: string,
    sessionId: string,
    userContext?: UserContext
  ): Promise<string> {
    const startTime = Date.now();
    const cleanQuery = query.replace(/\s+/g, ' ').trim();

    Logger.info(`🎯 [TieredAgent] Processing query: "${cleanQuery}"`);

    if (this.isSimpleTextQuery(cleanQuery)) {
      Logger.info(`⚡ [Tier1] Using direct API for text generation`);
      this.tierCounts.tier1++;
      return this.processDirectAPI(cleanQuery, userContext, startTime);
    }

    if (this.isSingleToolQuery(cleanQuery)) {
      Logger.info(`🔧 [Tier2] Using manual tool routing`);
      this.tierCounts.tier2++;
      return this.processManualToolRouting(cleanQuery, userContext, startTime);
    }

    Logger.info(`🤖 [Tier3] Using full agent for complex workflow`);
    this.tierCounts.tier3++;
    return this.processFullAgent(cleanQuery, sessionId, userContext, startTime);
  }

  private isSimpleTextQuery(query: string): boolean {
    const visionAnalysisPatterns = /(analyze.*screen|what.*(can you see|do you see|see|on).*screen|describe.*screen|tell me.*about.*screen|explain.*what.*(see|on))/i;
    if (visionAnalysisPatterns.test(query)) return false;

    const complexKeywords = [
      'analyze', 'analysis', 'comprehensive', 'detailed report',
      'research and', 'organize and', 'plan a', 'strategy',
      'workflow', 'debug', 'optimize', 'suggest', 'recommend',
      'create a detailed', 'multi-step', 'complex'
    ];
    if (complexKeywords.some(keyword => query.toLowerCase().includes(keyword))) return false;

    const textPatterns = [
      /^(write|create|generate|draft|compose)/i,
      /^(explain|describe|tell me about|what is|how does)/i,
      /^(help me (with|understand)|can you help)/i,
      /^(what|why|how|when|where|who)/i,
      /\?$/,
      /^(hi|hello|hey|good morning|good afternoon)/i,
      /^(thanks|thank you|thx)/i,
      /^(yes|no|okay|ok|sure)/i,
      /(email|letter|message|text|content|summary|report)/i,
      /(idea|suggestion|advice|recommendation)/i
    ];

    const actionKeywords = [
      'open', 'launch', 'start', 'run', 'execute',
      'screenshot', 'capture', 'take a picture',
      'find', 'search', 'locate', 'look for',
      'organize', 'move', 'delete', 'create folder',
      'install', 'update', 'download'
    ];
    if (actionKeywords.some(keyword => query.toLowerCase().includes(keyword))) return false;

    return textPatterns.some(pattern => pattern.test(query));
  }

  private isSingleToolQuery(query: string): boolean {
    const visionAnalysisPatterns = /(analyze.*screen|what.*(can you see|do you see|see|on).*screen|describe.*screen|tell me.*about.*screen|explain.*what.*(see|on))/i;
    if (visionAnalysisPatterns.test(query)) return false;

    const toolPatterns = {
      screenshot: /(take|capture|get|grab) .*(screenshot|screen capture|screen shot)/i,
      appLauncher: /(open|launch|start)\s+\S+/i,
      cli: /(run|execute|terminal|command|git|npm|brew|curl|ping)/i,
      fileSystem: /(find|search|locate|look for) .*(file|folder|document)/i
    };

    return Object.values(toolPatterns).some(pattern => pattern.test(query));
  }

  private async processDirectAPI(
    query: string,
    userContext?: UserContext,
    startTime?: number
  ): Promise<string> {
    try {
      let fullText = '';

      await this.llmService.streamResponse(query, [], {
        onToken: (token) => { fullText += token; },
        onComplete: (text) => { fullText = text; },
        onError: (err) => { throw err; }
      }, true);

      const processingTime = Date.now() - (startTime || Date.now());
      Logger.info(`⚡ [Tier1] Completed in ${processingTime}ms`);

      return fullText;
    } catch (error) {
      Logger.error('❌ [Tier1] LLM Service failed:', error);
      if (this.openaiKey) {
        return this.processFullAgent(query, `fallback-${Date.now()}`, userContext, startTime);
      }
      return "I encountered an error processing your request. Please check your AI settings.";
    }
  }

  private async processManualToolRouting(
    query: string,
    userContext?: UserContext,
    startTime?: number
  ): Promise<string> {
    try {
      if (/(take|capture|get|grab) .*(screenshot|screen capture|screen shot)/i.test(query)) {
        Logger.info(`📸 [Tier2] Routing to screenshot tool`);
        return await visionTool.func({ action: "capture", query: null });
      }

      if (/(open|launch|start)/i.test(query)) {
        Logger.info(`🚀 [Tier2] Routing to app launcher`);
        const appMatch = query.match(/(open|launch|start)\s+(.+?)(?:\s+(?:now|please|for me))?$/i);
        const appName = appMatch ? appMatch[2].trim() : query;
        return await appLauncherTool.func({ command: `open ${appName}`, directExecution: true });
      }

      if (/(run|execute|terminal|command)/i.test(query)) {
        Logger.info(`💻 [Tier2] Routing to CLI tool`);
        const commandMatch = query.match(/(run|execute)\s+(.*)/i);
        const command = commandMatch ? commandMatch[2] : query;
        return await cliTool.func({ command });
      }

      if (/(find|search|locate|look for)/i.test(query)) {
        Logger.info(`📁 [Tier2] Routing to file system tool`);
        return await fileSystemTool.func({ operation: 'list', filePath: os.homedir() });
      }

      if (this.openaiKey) {
        Logger.info(`🔄 [Tier2] No tool match, falling back to full agent`);
        return this.processFullAgent(query, `tier2-fallback-${Date.now()}`, userContext, startTime);
      } else {
        Logger.info(`🔄 [Tier2] No tool match, falling back to Tier 1`);
        return this.processDirectAPI(query, userContext, startTime);
      }
    } catch (error) {
      Logger.error('❌ [Tier2] Manual routing failed:', error);
      if (this.openaiKey) {
        return this.processFullAgent(query, `tier2-error-${Date.now()}`, userContext, startTime);
      }
      return this.processDirectAPI(query, userContext, startTime);
    }
  }

  private async processFullAgent(
    query: string,
    sessionId: string,
    userContext?: UserContext,
    startTime?: number
  ): Promise<string> {
    if (!this.openaiKey) {
      return this.processDirectAPI(query, userContext, startTime);
    }

    try {
      const result = await this.fullAgent.invoke(
        {
          messages: [new HumanMessage(query)]
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
      Logger.info(`🤖 [Tier3] Completed in ${processingTime}ms`);

      return response;
    } catch (error) {
      Logger.error('❌ [Tier3] Full agent failed:', error);
      return "I encountered an error processing your request. Please try again.";
    }
  }

  async clearMemory(sessionId?: string): Promise<void> {
    if (sessionId) {
      Logger.info(`🧹 [TieredAgent] Clearing memory for session: ${sessionId}`);
    } else {
      Logger.info(`🧹 [TieredAgent] Clearing all memory`);
      this.checkpointSaver = new MemorySaver();
      this.initializeFullAgent();
    }
  }

  getPerformanceStats(): { tier1Count: number; tier2Count: number; tier3Count: number } {
    return {
      tier1Count: this.tierCounts.tier1,
      tier2Count: this.tierCounts.tier2,
      tier3Count: this.tierCounts.tier3,
    };
  }
}
