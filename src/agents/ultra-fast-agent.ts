import { Logger } from "../core/logger";

type UserContext = { userId?: string; displayName?: string; email?: string };

/**
 * Ultra-lightweight Jarvis Agent - Zero LangGraph overhead
 * Expected performance: 1-2 seconds for most queries
 */
export class UltraFastJarvisAgent {
  private openaiKey: string;

  constructor(openaiKey: string) {
    this.openaiKey = openaiKey;
  }

  async processQuery(query: string, userContext?: UserContext): Promise<string> {
    const startTime = Date.now();
    Logger.info(`⚡ [UltraFast] Processing: "${query}"`);

    const route = this.routeQuery(query);

    try {
      let result: string;

      switch (route.type) {
        case 'app_launch':
          result = await this.launchApp(route.params.appName);
          break;

        case 'screenshot':
          result = await this.takeScreenshot();
          break;

        case 'text_generation':
        default:
          result = await this.generateText(query, userContext);
          break;
      }

      const time = Date.now() - startTime;
      Logger.info(`⚡ [UltraFast] Completed in ${time}ms`);
      return result;

    } catch (error) {
      Logger.error('❌ [UltraFast] Error:', error);
      return "I encountered an error. Please try again.";
    }
  }

  private routeQuery(query: string): { type: string; params: Record<string, any> } {
    const appPatterns = [
      { pattern: /open\s+(.+)/i, extract: (match: RegExpMatchArray) => match[1].trim() },
      { pattern: /launch\s+(.+)/i, extract: (match: RegExpMatchArray) => match[1].trim() },
      { pattern: /start\s+(.+)/i, extract: (match: RegExpMatchArray) => match[1].trim() }
    ];

    for (const { pattern, extract } of appPatterns) {
      const match = query.match(pattern);
      if (match) {
        return { type: 'app_launch', params: { appName: extract(match) } };
      }
    }

    if (query.toLowerCase().includes('screenshot') || query.toLowerCase().includes('capture screen')) {
      return { type: 'screenshot', params: {} };
    }

    return { type: 'text_generation', params: {} };
  }

  private async generateText(query: string, userContext?: UserContext): Promise<string> {
    const prompt = this.buildTextPrompt(query, userContext);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.openaiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are Jarvis. Be helpful and concise.' },
            { role: 'user', content: prompt }
          ],
          max_tokens: 512,
          temperature: 0.1
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.choices[0]?.message?.content || "I couldn't generate a response.";
    } finally {
      clearTimeout(timeout);
    }
  }

  private async launchApp(appName: string): Promise<string> {
    try {
      const { appLauncherService } = await import('../services/app-launcher-service');
      await appLauncherService.findAndLaunchApp(appName);
      return `Launching ${appName}...`;
    } catch (error) {
      return `Failed to launch ${appName}. ${String(error)}`;
    }
  }

  private async takeScreenshot(): Promise<string> {
    try {
      const { captureScreen } = await import('../tools/vision-tool');
      await captureScreen();
      return "Screenshot captured!";
    } catch (error) {
      return `Failed to take screenshot. ${String(error)}`;
    }
  }

  private buildTextPrompt(query: string, userContext?: UserContext): string {
    if (userContext?.displayName) {
      return `User: ${userContext.displayName}\nQuery: ${query}`;
    }
    return query;
  }
}
