export interface NovelVoiceCastAPI {
  platform: string
  versions: Readonly<{
    electron: string
    chrome: string
  }>
}
