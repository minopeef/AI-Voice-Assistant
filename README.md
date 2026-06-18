# AI Assistant

A desktop voice dictation and AI assistant for macOS. Hold a single key, speak,
and clean, punctuated text is typed into whatever application is in focus.
Transcription and language processing can run fully on-device or against cloud
providers, depending on your priorities for privacy, speed, and accuracy.

## Features

- Push-to-talk dictation: hold a key, speak, release, and the transcribed text
  is inserted at the current cursor position in any application.
- Automatic cleanup of filler words ("um", "like"), grammar correction, and
  optional rephrasing, summarising, or bullet-pointing of the spoken text.
- On-device transcription via local Whisper and Sherpa-ONNX models
  (Parakeet and SenseVoice).
- Cloud transcription via Deepgram and OpenAI for lower latency and higher
  accuracy.
- Local language-model support through Ollama, or cloud models through Gemini
  and OpenAI.
- Agent layer capable of running small actions and tool calls (launching
  applications, filesystem operations, command execution, screen vision).
- Fully customisable prompts controlling how dictation is cleaned and how the
  assistant formats its output.
- No telemetry and no tracking; local-only operation requires no network access.

## Architecture

The application is built on Electron with a React renderer and a TypeScript
main process, plus native Objective-C/C++ addons for low-level macOS
integration.

```
src/
  main.ts            Electron main-process entry point
  preload.ts         Context-isolated bridge to the renderer
  App.tsx            React renderer entry point
  core/              App initialisation, agent manager, LLM service/providers
  agents/            Tiered agent implementations (fast / streaming / unified)
  transcription/     Whisper, Sherpa-ONNX, Deepgram and OpenAI transcribers
  input/             Audio capture, session orchestration, output management
  tools/             Tool registry and individual tools (app launcher,
                     filesystem, CLI, vision, system info)
  vision/            Screen-capture and visual analysis
  services/          Command parsing, text enhancement, settings, notifications
  ipc/               Inter-process handlers (auth, chat, dictation, settings)
  server/            Local Fastify server
  native/            Native module bindings
packages/
  whisper-addon/     Native Whisper addon built on whisper.cpp
```

Native modules (compiled through node-gyp) provide audio capture, global key
monitoring, function-key detection, and synthetic typing.

## Transcription and model options

Two transcription paths are supported:

- Local: Whisper (tiny / base / small) or Sherpa-ONNX models such as Parakeet
  and SenseVoice. Runs entirely offline with no API keys.
- Cloud: Deepgram or OpenAI for streaming, low-latency transcription.

Language post-processing can likewise run locally through Ollama or against a
cloud provider (Gemini or OpenAI). Local processing typically adds a small
amount of latency on consumer hardware; cloud processing or disabling
post-processing keeps transcription effectively instant.

## Requirements

- macOS 10.13 or newer
- Node.js 18 or newer
- Xcode command-line tools (for building native modules)

## Build from source

```bash
npm install
npm run build
npm run dev
```

Useful scripts:

| Script | Purpose |
|--------|---------|
| `npm run dev` | Watch-mode build and launch in development |
| `npm run build` | Build native modules and bundle the application |
| `npm run server` | Run the local server |
| `npm run lint` | Lint the TypeScript sources |
| `npm run type-check` | Type-check without emitting |
| `npm test` | Run the test suite |
| `npm run build:dmg` | Produce distributable disk images |

## Testing

```bash
npm test                 # full suite
npm run test:watch       # watch mode
npm run test:coverage    # coverage report
npm run test:integration # integration tests
npm run test:regression  # regression tests
npm run test:performance # performance tests
```

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Hold the dictation key | Start / stop recording |
| Double-tap the dictation key | Toggle hands-free mode |
| Escape | Cancel recording |

## Privacy

The application is designed to run without sending data off the device. When
local transcription and a local language model are selected, no network
requests are required for core functionality. No usage analytics or telemetry
are collected.

## License

Released under the MIT License. See the LICENSE file for details.
