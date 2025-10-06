import { EventEmitter } from 'events';
import { AudioManager } from '../audio/AudioManager';
import { VoiceEngine } from '../voice/VoiceEngine';
import { VoiceAugmentation } from '../voice/VoiceAugmentation';
import { ConfigManager } from '../config/ConfigManager';
import { logger } from '../utils/logger';

export enum OrchestratorState {
  IDLE = 'IDLE',
  INITIALIZING = 'INITIALIZING',
  READY = 'READY',
  PROCESSING = 'PROCESSING',
  ERROR = 'ERROR',
  SHUTDOWN = 'SHUTDOWN'
}

export interface ProcessingResult {
  success: boolean;
  outputPath?: string;
  error?: string;
  duration?: number;
}

export class MiniMoonOrchestrator extends EventEmitter {
  private state: OrchestratorState = OrchestratorState.IDLE;
  private audioManager: AudioManager | null = null;
  private voiceEngine: VoiceEngine | null = null;
  private voiceAugmentation: VoiceAugmentation | null = null;
  private configManager: ConfigManager;
  private initializationPromise: Promise<void> | null = null;
  private isShuttingDown: boolean = false;

  constructor(configManager: ConfigManager) {
    super();
    this.configManager = configManager;
    logger.info('MiniMoonOrchestrator: Constructor initialized');
  }

  /**
   * Inicializa todos los componentes del orquestador
   */
  async initialize(): Promise<void> {
    // Prevenir múltiples inicializaciones concurrentes
    if (this.initializationPromise) {
      logger.warn('MiniMoonOrchestrator: Initialization already in progress');
      return this.initializationPromise;
    }

    if (this.state === OrchestratorState.READY) {
      logger.info('MiniMoonOrchestrator: Already initialized');
      return;
    }

    this.initializationPromise = this._initialize();
    return this.initializationPromise;
  }

  private async _initialize(): Promise<void> {
    try {
      this.setState(OrchestratorState.INITIALIZING);
      logger.info('MiniMoonOrchestrator: Starting initialization');

      // Inicializar componentes en orden de dependencias
      await this.initializeAudioManager();
      await this.initializeVoiceEngine();
      await this.initializeVoiceAugmentation();

      this.setState(OrchestratorState.READY);
      logger.info('MiniMoonOrchestrator: Initialization completed successfully');
      this.emit('ready');
    } catch (error) {
      this.setState(OrchestratorState.ERROR);
      logger.error('MiniMoonOrchestrator: Initialization failed', error);
      this.emit('error', error);
      throw error;
    } finally {
      this.initializationPromise = null;
    }
  }

  private async initializeAudioManager(): Promise<void> {
    try {
      logger.info('MiniMoonOrchestrator: Initializing AudioManager');
      this.audioManager = new AudioManager(this.configManager);
      await this.audioManager.initialize();
      logger.info('MiniMoonOrchestrator: AudioManager initialized');
    } catch (error) {
      logger.error('MiniMoonOrchestrator: AudioManager initialization failed', error);
      throw new Error(`AudioManager initialization failed: ${error}`);
    }
  }

  private async initializeVoiceEngine(): Promise<void> {
    try {
      logger.info('MiniMoonOrchestrator: Initializing VoiceEngine');
      this.voiceEngine = new VoiceEngine(this.configManager);
      await this.voiceEngine.initialize();
      logger.info('MiniMoonOrchestrator: VoiceEngine initialized');
    } catch (error) {
      logger.error('MiniMoonOrchestrator: VoiceEngine initialization failed', error);
      throw new Error(`VoiceEngine initialization failed: ${error}`);
    }
  }

  private async initializeVoiceAugmentation(): Promise<void> {
    try {
      logger.info('MiniMoonOrchestrator: Initializing VoiceAugmentation');
      this.voiceAugmentation = new VoiceAugmentation(this.configManager);
      await this.voiceAugmentation.initialize();
      logger.info('MiniMoonOrchestrator: VoiceAugmentation initialized');
    } catch (error) {
      logger.error('MiniMoonOrchestrator: VoiceAugmentation initialization failed', error);
      throw new Error(`VoiceAugmentation initialization failed: ${error}`);
    }
  }

  /**
   * Procesa un archivo de audio con voz sintética
   */
  async processAudioFile(
    inputPath: string,
    outputPath: string,
    options: {
      voiceProfile?: string;
      augmentationLevel?: number;
      preserveOriginal?: boolean;
    } = {}
  ): Promise<ProcessingResult> {
    const startTime = Date.now();

    try {
      this.validateReady();
      this.setState(OrchestratorState.PROCESSING);
      
      logger.info('MiniMoonOrchestrator: Starting audio processing', {
        inputPath,
        outputPath,
        options
      });

      // Cargar audio - Método correcto
      const audioBuffer = await this.audioManager!.loadAudioFile(inputPath);
      this.emit('progress', { stage: 'loaded', progress: 0.2 });

      // Generar voz sintética - Método correcto
      const voiceBuffer = await this.voiceEngine!.generateVoice(audioBuffer, {
        voiceId: options.voiceProfile
      });
      this.emit('progress', { stage: 'synthesized', progress: 0.5 });

      // Aplicar augmentación - Método correcto
      const augmentedBuffer = await this.voiceAugmentation!.applyAugmentation(voiceBuffer, {
        intensity: options.augmentationLevel || 0.7
      });
      this.emit('progress', { stage: 'augmented', progress: 0.8 });

      // Guardar resultado - Método correcto
      await this.audioManager!.saveAudioFile(augmentedBuffer, outputPath);
      this.emit('progress', { stage: 'saved', progress: 1.0 });

      const duration = Date.now() - startTime;
      logger.info('MiniMoonOrchestrator: Processing completed', {
        duration,
        outputPath
      });

      this.setState(OrchestratorState.READY);
      
      return {
        success: true,
        outputPath,
        duration
      };
    } catch (error) {
      logger.error('MiniMoonOrchestrator: Processing failed', error);
      this.setState(OrchestratorState.READY); // No ERROR, volver a READY
      this.emit('error', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Procesa texto a voz directamente
   */
  async synthesizeText(
    text: string,
    outputPath: string,
    options: {
      voiceProfile?: string;
      augmentationLevel?: number;
    } = {}
  ): Promise<ProcessingResult> {
    const startTime = Date.now();

    try {
      this.validateReady();
      this.setState(OrchestratorState.PROCESSING);

      logger.info('MiniMoonOrchestrator: Starting text synthesis', {
        textLength: text.length,
        outputPath,
        options
      });

      // Generar voz desde texto - Método correcto
      const voiceBuffer = await this.voiceEngine!.synthesizeFromText(text, {
        voiceId: options.voiceProfile
      });
      this.emit('progress', { stage: 'synthesized', progress: 0.6 });

      // Aplicar augmentación - Método correcto
      const augmentedBuffer = await this.voiceAugmentation!.applyAugmentation(voiceBuffer, {
        intensity: options.augmentationLevel || 0.7
      });
      this.emit('progress', { stage: 'augmented', progress: 0.9 });

      // Guardar resultado - Método correcto
      await this.audioManager!.saveAudioFile(augmentedBuffer, outputPath);
      this.emit('progress', { stage: 'saved', progress: 1.0 });

      const duration = Date.now() - startTime;
      logger.info('MiniMoonOrchestrator: Synthesis completed', {
        duration,
        outputPath
      });

      this.setState(OrchestratorState.READY);

      return {
        success: true,
        outputPath,
        duration
      };
    } catch (error) {
      logger.error('MiniMoonOrchestrator: Synthesis failed', error);
      this.setState(OrchestratorState.READY); // No ERROR, volver a READY
      this.emit('error', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Valida que el orquestador esté listo para procesar
   */
  private validateReady(): void {
    if (this.isShuttingDown) {
      throw new Error('Orchestrator is shutting down');
    }

    if (this.state === OrchestratorState.IDLE) {
      throw new Error('Orchestrator not initialized. Call initialize() first.');
    }

    if (this.state === OrchestratorState.ERROR) {
      throw new Error('Orchestrator is in error state. Reinitialize required.');
    }

    if (this.state === OrchestratorState.PROCESSING) {
      throw new Error('Orchestrator is already processing. Wait for completion.');
    }

    if (!this.audioManager || !this.voiceEngine || !this.voiceAugmentation) {
      throw new Error('One or more components not initialized');
    }
  }

  /**
   * Cambia el estado del orquestador
   */
  private setState(newState: OrchestratorState): void {
    const oldState = this.state;
    this.state = newState;
    logger.debug('MiniMoonOrchestrator: State changed', {
      from: oldState,
      to: newState
    });
    this.emit('stateChange', { from: oldState, to: newState });
  }

  /**
   * Obtiene el estado actual
   */
  getState(): OrchestratorState {
    return this.state;
  }

  /**
   * Verifica si está listo para procesar
   */
  isReady(): boolean {
    return this.state === OrchestratorState.READY;
  }

  /**
   * Limpia y cierra todos los recursos
   */
  async shutdown(): Promise<void> {
    if (this.isShuttingDown || this.state === OrchestratorState.SHUTDOWN) {
      logger.warn('MiniMoonOrchestrator: Already shutting down or shut down');
      return;
    }

    try {
      this.isShuttingDown = true;
      logger.info('MiniMoonOrchestrator: Starting shutdown');

      // Esperar a que termine el procesamiento actual con timeout
      if (this.state === OrchestratorState.PROCESSING) {
        logger.info('MiniMoonOrchestrator: Waiting for current processing to finish');
        await Promise.race([
          new Promise<void>(resolve => {
            const checkInterval = setInterval(() => {
              if (this.state !== OrchestratorState.PROCESSING) {
                clearInterval(checkInterval);
                resolve();
              }
            }, 100);
          }),
          new Promise<void>(resolve => setTimeout(resolve, 5000)) // timeout 5s
        ]);
      }

      // Limpiar componentes en orden inverso
      if (this.voiceAugmentation) {
        await this.voiceAugmentation.dispose();
        this.voiceAugmentation = null;
      }

      if (this.voiceEngine) {
        await this.voiceEngine.dispose();
        this.voiceEngine = null;
      }

      if (this.audioManager) {
        await this.audioManager.dispose();
        this.audioManager = null;
      }

      this.setState(OrchestratorState.SHUTDOWN);
      this.removeAllListeners();
      
      logger.info('MiniMoonOrchestrator: Shutdown completed');
    } catch (error) {
      logger.error('MiniMoonOrchestrator: Shutdown error', error);
      throw error;
    } finally {
      this.isShuttingDown = false;
    }
  }

  /**
   * Obtiene información del estado de los componentes
   */
  getComponentsStatus(): {
    audioManager: boolean;
    voiceEngine: boolean;
    voiceAugmentation: boolean;
  } {
    return {
      audioManager: this.audioManager !== null,
      voiceEngine: this.voiceEngine !== null,
      voiceAugmentation: this.voiceAugmentation !== null
    };
  }
}
