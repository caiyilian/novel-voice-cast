import type { NovelVoiceCastAPI } from "../preload/types"

declare global {
  interface Window {
    novelVoiceCast: NovelVoiceCastAPI
  }
}

export {}
