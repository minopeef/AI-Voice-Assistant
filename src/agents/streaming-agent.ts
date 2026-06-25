import { Logger } from "../core/logger";

type UserContext = { userId?: string; displayName?: string; email?: string };
type RouteResult = { type: 'text_generation' | 'app_launch' | 'screenshot'; params: Record<string, any> };

/**
 * Streaming Jarvis Agent - Provides instant feedback
 * User sees response immediately as it's generated
 */
export class StreamingJarvisAgent {
  private openaiKey: string;

  constructor(openaiKey: string) {
    this.openaiKey = openaiKey;
  }

  async processQueryStreaming(
    query: string,
    onChunk: (chunk: string) => void,
    userContext?: UserContext
  ): Promise<void> {
    Logger.info(`🌊 [Streaming] Processing: "${query}"`);

    try {
      const route = this.routeQuery(query);

      if (route.type === 'text_generation') {
        await this.streamTextGeneration(query, onChunk, userContext);
      } else {
        onChunk(`Processing ${route.type}...\n`);
        const result = await this.executeNonStreamingTask(route);
        onChunk(result);
      }
    } catch (error) {
      onChunk(`Error: ${String(error)}`);
    }
  }

  private async streamTextGeneration(
    query: string,
    onChunk: (chunk: string) => void,
    userContext?: UserContext
  ): Promise<void> {
    const systemContent = userContext?.displayName
      ? `You are Jarvis. Be helpful and concise. User: ${userContext.displayName}.`
      : 'You are Jarvis. Be helpful and concise.';

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.openaiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemContent },
          { role: 'user', content: query }
        ],
        max_tokens: 512,
        temperature: 0.1,
        stream: true
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;

          const data = trimmed.slice(6);
          if (data === '[DONE]') return;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) {
              onChunk(content);
            }
          } catch {
            // Skip unparseable SSE lines
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private routeQuery(query: string): RouteResult {
    const lower = query.toLowerCase();

    if (lower.includes('open') || lower.includes('launch')) {
      return { type: 'app_launch', params: { query } };
    }

    if (lower.includes('screenshot')) {
      return { type: 'screenshot', params: {} };
    }

    return { type: 'text_generation', params: {} };
  }

  private async executeNonStreamingTask(route: RouteResult): Promise<string> {
    switch (route.type) {
      case 'app_launch': {
        try {
          const { appLauncherService } = await import('../services/app-launcher-service');
          const appMatch = (route.params.query as string).match(/(open|launch)\s+(.+)/i);
          const appName = appMatch ? appMatch[2].trim() : route.params.query as string;
          return await appLauncherService.launchApp(appName);
        } catch (error) {
          return `Failed to launch app: ${String(error)}`;
        }
      }
      case 'screenshot': {
        try {
          const { captureScreen } = await import('../tools/vision-tool');
          return await captureScreen();
        } catch (error) {
          return `Failed to capture screenshot: ${String(error)}`;
        }
      }
      default:
        return "Task completed!";
    }
  }
}
