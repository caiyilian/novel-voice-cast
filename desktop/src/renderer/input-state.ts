import type {
  InputKind,
  SelectedTextFile,
  TextFileSelection,
} from "../preload/types"

export interface InputSlotState {
  file: SelectedTextFile | null
  error: string
}

export type InputState = Record<InputKind, InputSlotState>

export function emptyInputState(): InputState {
  return {
    novel: { file: null, error: "" },
    labels: { file: null, error: "" },
  }
}

export function applySelection(
  state: InputState,
  inputKind: InputKind,
  selection: TextFileSelection | null,
): InputState {
  if (selection === null) return state
  return {
    ...state,
    [inputKind]: selection.ok
      ? { file: selection.file, error: "" }
      : { file: null, error: selection.error },
  }
}

export function clearSelection(state: InputState, inputKind: InputKind): InputState {
  return { ...state, [inputKind]: { file: null, error: "" } }
}

export function inputsReady(state: InputState): boolean {
  return Boolean(state.novel.file && state.labels.file)
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
