/**
 * Minimal WAV reading and joining.
 *
 * Long answers are synthesised as several independent clips so they can be
 * generated in parallel — see `tts.service`. Each clip comes back as a complete
 * RIFF file, so they cannot simply be concatenated: the result would carry a
 * header in the middle of the audio. Joining means taking the PCM out of each
 * and writing one fresh header over the total.
 */

const HEADER_BYTES = 44;

export interface WavAudio {
  pcm: Buffer;
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
}

/**
 * Pull the PCM and format out of a RIFF/WAVE buffer.
 *
 * The chunk list is walked rather than assuming `data` sits at the canonical
 * offset 44 — an encoder that emits a `LIST` or `fact` chunk first would
 * otherwise have its metadata played as a burst of noise.
 */
export function parseWav(buffer: Buffer): WavAudio | null {
  if (buffer.length < 12) return null;
  if (buffer.toString('ascii', 0, 4) !== 'RIFF') return null;
  if (buffer.toString('ascii', 8, 12) !== 'WAVE') return null;

  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;

  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const body = offset + 8;

    if (id === 'fmt ' && body + 16 <= buffer.length) {
      channels = buffer.readUInt16LE(body + 2);
      sampleRate = buffer.readUInt32LE(body + 4);
      bitsPerSample = buffer.readUInt16LE(body + 14);
    } else if (id === 'data') {
      // Trust the buffer over the declared size: a truncated response would
      // otherwise slice past the end.
      const end = Math.min(body + size, buffer.length);
      if (!sampleRate || !channels || !bitsPerSample) return null;
      return { pcm: buffer.subarray(body, end), sampleRate, channels, bitsPerSample };
    }

    // RIFF chunks are word-aligned; odd sizes carry a pad byte.
    offset = body + size + (size % 2);
  }

  return null;
}

/** Write a canonical 44-byte PCM WAV header in front of `pcm`. */
export function encodeWav(audio: WavAudio): Buffer {
  const { pcm, sampleRate, channels, bitsPerSample } = audio;
  const byteRate = (sampleRate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const header = Buffer.alloc(HEADER_BYTES);

  header.write('RIFF', 0, 'ascii');
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write('WAVE', 8, 'ascii');
  header.write('fmt ', 12, 'ascii');
  header.writeUInt32LE(16, 16); // PCM fmt chunk size
  header.writeUInt16LE(1, 20); // audio format: PCM
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36, 'ascii');
  header.writeUInt32LE(pcm.length, 40);

  return Buffer.concat([header, pcm]);
}

/**
 * Join WAV clips into one, in order.
 *
 * Returns the single input unchanged when there is nothing to join, and `null`
 * if any clip is unparseable or the formats disagree — callers fall back to
 * playing the first clip alone rather than emitting a garbled file.
 */
export function concatWav(clips: Buffer[]): Buffer | null {
  if (clips.length === 0) return null;
  if (clips.length === 1) return clips[0] ?? null;

  const parsed: WavAudio[] = [];
  for (const clip of clips) {
    const audio = parseWav(clip);
    if (!audio) return null;
    parsed.push(audio);
  }

  const [first] = parsed;
  if (!first) return null;

  const uniform = parsed.every(
    (a) =>
      a.sampleRate === first.sampleRate &&
      a.channels === first.channels &&
      a.bitsPerSample === first.bitsPerSample,
  );
  if (!uniform) return null;

  return encodeWav({
    pcm: Buffer.concat(parsed.map((a) => a.pcm)),
    sampleRate: first.sampleRate,
    channels: first.channels,
    bitsPerSample: first.bitsPerSample,
  });
}
