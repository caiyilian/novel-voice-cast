export type InputKind = "novel" | "labels"

export interface SelectedTextFile {
  path: string
  name: string
  size: number
  modifiedAt: string
}

export type TextFileSelection =
  | { ok: true; file: SelectedTextFile }
  | { ok: false; error: string }

export interface NovelVoiceCastAPI {
  platform: string
  versions: Readonly<{
    electron: string
    chrome: string
  }>
  pickTextFile: (inputKind: InputKind) => Promise<TextFileSelection | null>
  acceptDroppedTextFile: (file: File, inputKind: InputKind) => Promise<TextFileSelection>
}
