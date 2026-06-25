import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Logger } from "../core/logger";

type UserContext = { userId?: string; displayName?: string; email?: string };

/**
 * Ultra-fast Jarvis Agent using OpenAI function calling instead of LangGraph
 * Expected performance: 1-3 seconds vs 9+ seconds with LangGraph
 */
export class FastJarvisAgent {
  private llm: ChatOpenAI;
  private openaiKey: string;

  constructor(openaiKey: string) {
    this.openaiKey = openaiKey;
    this.llm = new ChatOpenAI({
      apiKey: openaiKey,
      model: "gpt-4o-mini",
      temperature: 0.1,
      maxTokens: 512,
      timeout: 3000,
      maxRetries: 1
    });
  }

  async processQuery(query: string, userContext?: UserContext): Promise<string> {
    const startTime = Date.now();
    Logger.info(`⚡ [FastAgent] Processing: "${query}"`);

    try {
      if (this.isSimpleTextQuery(query)) {
        return await this.handleTextQuery(query, userContext, startTime);
      }
      return await this.handleToolQuery(query, userContext, startTime);
    } catch (error) {
      Logger.error('❌ [FastAgent] Error:', error);
      return "I encountered an error. Please try again.";
    }
  }

  private isSimpleTextQuery(query: string): boolean {
    const toolKeywords = ['open', 'launch', 'screenshot', 'file', 'terminal', 'run'];
    return !toolKeywords.some(keyword => query.toLowerCase().includes(keyword));
  }

  private async handleTextQuery(query: string, userContext: UserContext | undefined, startTime: number): Promise<string> {
    const systemPrompt = `You are Jarvis. Be helpful and concise.${
      userContext?.displayName ? ` User: ${userContext.displayName}.` : ''
    }`;

    const prompt = this.buildTextPrompt(query, userContext);

    const response = await this.llm.invoke([
      new SystemMessage(systemPrompt),
      new HumanMessage(prompt)
    ]);

    const result = response.content as string;
    const time = Date.now() - startTime;
    Logger.info(`⚡ [FastAgent] Text query completed in ${time}ms`);
    return result;
  }

  private async handleToolQuery(query: string, userContext: UserContext | undefined, startTime: number): Promise<string> {
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "launch_app",
          description: "Open or launch applications",
          parameters: {
            type: "object",
            properties: {
              app_name: { type: "string", description: "Name of app to launch" }
            },
            required: ["app_name"]
          }
        }
      },
      {
        type: "function" as const,
        function: {
          name: "take_screenshot",
          description: "Take a screenshot and analyze it",
          parameters: { type: "object", properties: {}, required: [] }
        }
      }
    ];

    const response = await this.llm.invoke([
      new SystemMessage("You are Jarvis. Use the available functions to help the user."),
      new HumanMessage(query)
    ], {
      tools,
      tool_choice: "auto"
    });

    if (response.tool_calls && response.tool_calls.length > 0) {
      const toolCall = response.tool_calls[0];
      const result = await this.executeFunction(toolCall.name, JSON.stringify(toolCall.args));

      const time = Date.now() - startTime;
      Logger.info(`⚡ [FastAgent] Tool query completed in ${time}ms`);
      return result;
    }

    const time = Date.now() - startTime;
    Logger.info(`⚡ [FastAgent] Fallback text response in ${time}ms`);
    return response.content as string;
  }

  private async executeFunction(name: string, args: string): Promise<string> {
    let params: Record<string, any>;
    try {
      params = JSON.parse(args);
    } catch {
      Logger.error(`❌ [FastAgent] Failed to parse function args for ${name}:`, args);
      return "Failed to execute function due to invalid arguments.";
    }

    switch (name) {
      case "launch_app": {
        const { appLauncherService } = await import("../services/app-launcher-service");
        return await appLauncherService.launchApp(params.app_name);
      }
      case "take_screenshot": {
        const { captureScreen } = await import("../tools/vision-tool");
        return await captureScreen();
      }
      default:
        return "Function not implemented";
    }
  }

  private buildTextPrompt(query: string, userContext?: UserContext): string {
    if (userContext?.displayName) {
      return `User: ${userContext.displayName}\nQuery: ${query}`;
    }
    return query;
  }
}
