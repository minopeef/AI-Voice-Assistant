import { TranscriptionSession, UserStats } from '../types/analytics';
import { LocalAnalyticsStore } from '../storage/local-analytics-store';
import { TimeSavingsCalculator } from '../services/time-savings-calculator';
import { Logger } from '../core/logger';
import { EventEmitter } from 'events';
import { posthog } from './posthog';

// Extends UserStats with the internal audio accumulator used for WPM calculation.
// This field is never persisted or exposed to callers.
interface CachedUserStats extends UserStats {
  _totalAudioMs: number;
}

/**
 * Optimized Analytics Manager with real-time updates and efficient queries
 */
export class OptimizedAnalyticsManager extends EventEmitter {
  private storage = new LocalAnalyticsStore();
  private currentSession: Partial<TranscriptionSession> | null = null;
  private userId: string = 'default-user';

  private cachedStats: CachedUserStats | null = null;
  private pendingUpdates: {
    sessions: number;
    words: number;
    characters: number;
    timeSaved: number;
    audioMs: number;
  } = { sessions: 0, words: 0, characters: 0, timeSaved: 0, audioMs: 0 };

  private updateTimer: NodeJS.Timeout | null = null;
  private readonly BATCH_UPDATE_DELAY = 5000;

  private lastEmitTime = 0;
  private readonly EMIT_THROTTLE_MS = 500;
  private emitTimeout: NodeJS.Timeout | null = null;

  constructor() {
    super();
  }

  setUserId(userId: string): void {
    Logger.debug('Setting userId from', this.userId, 'to', userId);

    if (this.userId !== userId) {
      this.cachedStats = null;
      this.pendingUpdates = { sessions: 0, words: 0, characters: 0, timeSaved: 0, audioMs: 0 };
      if (this.updateTimer) {
        clearTimeout(this.updateTimer);
        this.updateTimer = null;
      }
      if (this.emitTimeout) {
        clearTimeout(this.emitTimeout);
        this.emitTimeout = null;
      }
    }

    this.userId = userId;
    this.storage.setUserId(userId);
  }

  getCurrentUserId(): string {
    return this.userId;
  }

  startSession(): string {
    const sessionId = Date.now().toString();

    if (this.currentSession) {
      Logger.warning(`📊 [Analytics] WARNING: Starting new session ${sessionId} while session ${this.currentSession.id} is still active!`);
    }

    this.currentSession = {
      id: sessionId,
      startTime: new Date(),
      contextType: 'other'
    };
    Logger.info(`🔥 Session started: ${sessionId}, userId: ${this.userId}`);
    return sessionId;
  }

  async endSession(
    transcriptionText: string,
    audioLengthMs: number,
    model: string = 'whisper-local',
    mode: 'dictation' | 'command' = 'dictation'
  ): Promise<void> {
    Logger.info(`📊 [Analytics] endSession called - sessionId: ${this.currentSession?.id}, userId: ${this.userId}`);

    if (!this.currentSession) {
      Logger.warning('📊 [Analytics] No current session to end');
      return;
    }

    const wordCount = transcriptionText.trim().split(/\s+/).length;
    const characterCount = transcriptionText.length;
    const processingTimeMs = Date.now() - this.currentSession.startTime!.getTime();

    const session: TranscriptionSession = {
      ...this.currentSession,
      userId: this.userId,
      endTime: new Date(),
      transcriptionText,
      wordCount,
      characterCount,
      processingTimeMs,
      mode,
      metadata: {
        audioLengthMs,
        model,
        language: 'en'
      },
      createdAt: new Date()
    } as TranscriptionSession;

    const sessionSavings = TimeSavingsCalculator.calculateSessionSavings(session);
    const timeSaved = sessionSavings.sessionTimeSaved;

    this.pendingUpdates.sessions += 1;
    this.pendingUpdates.words += wordCount;
    this.pendingUpdates.characters += characterCount;
    this.pendingUpdates.timeSaved += timeSaved;
    this.pendingUpdates.audioMs += audioLengthMs;

    const sessionWPM = audioLengthMs > 0 ? Math.round((wordCount / audioLengthMs) * 60000) : 0;

    if (!this.cachedStats) {
      this.cachedStats = {
        userId: this.userId,
        totalSessions: 0,
        totalWords: 0,
        totalCharacters: 0,
        averageWPM: 0,
        estimatedTimeSavedMs: 0,
        lastActiveDate: new Date(),
        streakDays: 0,
        createdAt: new Date(),
        _totalAudioMs: 0
      };
      Logger.info('📊 [Analytics] Initialized default cached stats for real-time updates');
    }

    this.cachedStats.totalSessions += 1;
    this.cachedStats.totalWords += wordCount;
    this.cachedStats.totalCharacters += characterCount;
    this.cachedStats.estimatedTimeSavedMs += timeSaved;
    this.cachedStats._totalAudioMs += audioLengthMs;

    if (this.cachedStats._totalAudioMs > 0) {
      this.cachedStats.averageWPM = Math.round(
        (this.cachedStats.totalWords / this.cachedStats._totalAudioMs) * 60000
      );
    }

    Logger.info(`📊 [Analytics] Session WPM: ${sessionWPM}, Average WPM: ${this.cachedStats.averageWPM}`);

    this.emitStatsUpdate();

    Logger.info(`📊 [Analytics] Saving session locally - sessionId: ${session.id}, words: ${wordCount}`);
    this.storage.saveSession(session);

    // Anonymous usage pulse — never includes transcript text. No-op in
    // open-source builds (no key) or when the user has turned the
    // `analytics` toggle off in Settings → Privacy.
    posthog.capture('dictation_completed', {
      word_count: wordCount,
      character_count: characterCount,
      audio_length_ms: audioLengthMs,
      processing_time_ms: processingTimeMs,
      model,
      mode
    });

    this.scheduleBatchUpdate();

    const endedSessionId = this.currentSession.id;
    this.currentSession = null;
    Logger.info(`📊 [Analytics] Session ${endedSessionId} ended successfully - totalSessions now: ${this.cachedStats.totalSessions}`);
    Logger.info(`📊 [Analytics] Event listeners for stats-update: ${this.listenerCount('stats-update')}`);
  }

  private scheduleBatchUpdate(): void {
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
    }
    this.updateTimer = setTimeout(() => {
      this.performBatchUpdate().catch(err =>
        Logger.error('📊 [Analytics] Scheduled batch update failed:', err)
      );
    }, this.BATCH_UPDATE_DELAY);
  }

  private async performBatchUpdate(): Promise<void> {
    if (this.pendingUpdates.sessions === 0) return;

    Logger.info('📊 Performing batch stats update:', this.pendingUpdates);

    try {
      await this.storage.updateStatsInBatch(this.pendingUpdates);
      this.pendingUpdates = { sessions: 0, words: 0, characters: 0, timeSaved: 0, audioMs: 0 };
    } catch (error) {
      Logger.error('Failed to perform batch update:', error);
      // Keep pending updates for retry on next scheduled run
    }
  }

  private emitStatsUpdate(): void {
    const now = Date.now();
    const timeSinceLastEmit = now - this.lastEmitTime;

    if (timeSinceLastEmit >= this.EMIT_THROTTLE_MS) {
      this.performEmit();
    } else if (!this.emitTimeout) {
      const delay = this.EMIT_THROTTLE_MS - timeSinceLastEmit;
      this.emitTimeout = setTimeout(() => this.performEmit(), delay);
    }
  }

  private performEmit(): void {
    if (this.emitTimeout) {
      clearTimeout(this.emitTimeout);
      this.emitTimeout = null;
    }
    this.lastEmitTime = Date.now();
    Logger.info(`📊 [Analytics] Emitting stats-update (listeners: ${this.listenerCount('stats-update')})`);
    this.emit('stats-update', this.cachedStats);
  }

  async getStats(): Promise<UserStats | null> {
    if (this.cachedStats) {
      Logger.info(`📊 Returning cached stats - sessions: ${this.cachedStats.totalSessions}, userId: ${this.userId}`);
      return this.cachedStats;
    }

    Logger.info(`🔍 Loading stats from local storage for userId: ${this.userId}...`);
    const baseStats = await this.storage.getStats();
    if (!baseStats) {
      Logger.warning('📊 [Analytics] No stats in local storage - returning default stats');
      return {
        userId: this.userId,
        totalSessions: 0,
        totalWords: 0,
        totalCharacters: 0,
        averageWPM: 0,
        estimatedTimeSavedMs: 0,
        lastActiveDate: new Date(),
        streakDays: 0,
        createdAt: new Date()
      };
    }

    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const recentSessions = await this.storage.getUserSessionsInDateRange(
      this.userId,
      thirtyDaysAgo,
      new Date()
    );

    if (recentSessions && recentSessions.length > 0) {
      const recentSavings = TimeSavingsCalculator.calculateCumulativeSavings(recentSessions);

      // Use lifetime audio time for WPM so numerator and denominator share
      // the same window. _totalAudioMs is populated by LocalAnalyticsStore.toUserStats;
      // falls back to the recent-30d sum for legacy stores not yet backfilled.
      const lifetimeAudioMs = (baseStats as any)._totalAudioMs || 0;
      const recentAudioMs = recentSessions.reduce(
        (sum, s) => sum + (s.metadata?.audioLengthMs || 0), 0
      );
      const denomAudioMs = lifetimeAudioMs > 0 ? lifetimeAudioMs : recentAudioMs;
      const calculatedWPM = denomAudioMs > 0
        ? Math.round((baseStats.totalWords / denomAudioMs) * 60_000)
        : 0;

      this.cachedStats = {
        ...baseStats,
        averageWPM: calculatedWPM,
        dailyTimeSaved: recentSavings.dailySavings,
        weeklyTimeSaved: recentSavings.weeklySavings,
        monthlyTimeSaved: recentSavings.monthlySavings,
        efficiencyMultiplier: recentSavings.averageEfficiency,
        _totalAudioMs: denomAudioMs
      };
    } else {
      this.cachedStats = { ...baseStats, _totalAudioMs: (baseStats as any)._totalAudioMs || 0 };
    }

    return this.cachedStats;
  }

  async flush(): Promise<void> {
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
      this.updateTimer = null;
    }

    if (this.pendingUpdates.sessions > 0) {
      await this.performBatchUpdate();
    }

    await posthog.shutdown();
  }

  onStatsUpdate(callback: (stats: UserStats) => void): () => void {
    this.on('stats-update', callback);
    return () => this.off('stats-update', callback);
  }

  forceRefreshStats(): void {
    this.cachedStats = null;
  }

  clearState(): void {
    this.userId = 'default-user';
    this.currentSession = null;
    this.cachedStats = null;
    this.pendingUpdates = { sessions: 0, words: 0, characters: 0, timeSaved: 0, audioMs: 0 };
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
      this.updateTimer = null;
    }
    this.storage.setUserId('default-user');
  }

  trackEvent(eventName: string, properties?: Record<string, unknown>): void {
    Logger.info(`[Analytics] Event: ${eventName}`, properties);
    this.storage.trackEvent(eventName, properties);
  }

  trackError(errorType: string, properties?: Record<string, unknown>): void {
    Logger.error(`[Analytics] Error: ${errorType}`, properties);
    this.storage.trackEvent('error_' + errorType, properties);
  }

  trackPerformance(metricName: string, value: number, properties?: Record<string, unknown>): void {
    Logger.info(`[Analytics] Performance: ${metricName} = ${value}ms`, properties);
    this.storage.trackEvent('performance_' + metricName, { ...properties, value });
  }
}
