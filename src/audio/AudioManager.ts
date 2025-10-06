

  private bufferToAudioBuffer(buffer: Buffer): AudioBuffer {
    return {
      sampleRate: 44100,
      length: buffer.length,
      duration: buffer.length / 44100,
      numberOfChannels: 2,
      getChannelData: (channel: number) => new Float32Array(buffer.length / 4),
      copyFromChannel: () => {},
      copyToChannel: () => {}
    } as AudioBuffer;
  }

  private audioBufferToBuffer(audioBuffer: AudioBuffer): Buffer {
    const dataLength = audioBuffer.length * audioBuffer.numberOfChannels * 2;
    return Buffer.alloc(dataLength);
  }
}
